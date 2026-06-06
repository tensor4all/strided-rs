#!/usr/bin/env python3
"""PyTorch reference runner for strided-kernel mul benchmarks.

The timed loop uses torch.mul(..., out=...) so allocator and autograd overheads
are excluded. Shapes follow the original tenferro-benchmark row-major einsum
instances, while the Rust runner uses the equivalent strided-kernel column-major
views.
"""

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

ELEMENTWISE = "elementwise"
OUTER_PRODUCT = "outer_product"
BATCHED_OUTER_COMPACT = "batched_outer_compact"
BATCHED_OUTER_NONCOMPACT = "batched_outer_noncompact"
BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT = "batched_outer_noncompact_torchlike_output"
BATCHED_OUTER_NONCOMPACT_LHS_SCALAR = "batched_outer_noncompact_lhs_scalar"
BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP = "batched_outer_noncompact_single_outer_group"
PERMUTED_ELEMENTWISE = "permuted_elementwise"


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
            (ELEMENTWISE, (64,)),
            (OUTER_PRODUCT, (64,)),
            (BATCHED_OUTER_COMPACT, (4, 4, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT, (4, 4, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT, (4, 4, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT_LHS_SCALAR, (4, 4, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT, (2, 8, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT, (8, 2, 8, 8)),
            (BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP, (32, 32, 1, 1)),
            (PERMUTED_ELEMENTWISE, (6, 3)),
        ]
    if profile == "quick":
        return [
            (ELEMENTWISE, (1024,)),
            (OUTER_PRODUCT, (2048,)),
            (BATCHED_OUTER_COMPACT, (16, 16, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT, (16, 16, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT, (16, 16, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT_LHS_SCALAR, (16, 16, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT, (8, 32, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT, (32, 8, 64, 64)),
            (BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP, (256, 256, 1, 1)),
            (PERMUTED_ELEMENTWISE, (12, 3)),
        ]
    return [
        (ELEMENTWISE, (2048,)),
        (OUTER_PRODUCT, (4096,)),
        (BATCHED_OUTER_COMPACT, (16, 16, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT, (16, 16, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT, (16, 16, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT_LHS_SCALAR, (16, 16, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT, (8, 32, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT, (32, 8, 64, 64)),
        (BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP, (1024, 1024, 1, 1)),
        (PERMUTED_ELEMENTWISE, (16, 3)),
    ]


def benchmark_name(kind: str, dims: tuple[int, ...]) -> str:
    if kind == ELEMENTWISE:
        n = dims[0]
        return f"bin_elementwise_mul_{n}x{n}"
    if kind == OUTER_PRODUCT:
        return f"bin_outer_product_{dims[0]}"
    if kind == PERMUTED_ELEMENTWISE:
        rank, extent = dims
        return f"bin_permuted_elementwise_mul_rank{rank}_extent{extent}"

    j, k, o, t = dims
    if kind == BATCHED_OUTER_COMPACT:
        return f"bin_batched_outer_product_compact_j{j}_k{k}_o{o}_t{t}"
    if kind == BATCHED_OUTER_NONCOMPACT:
        return f"bin_batched_outer_product_noncompact_j{j}_k{k}_o{o}_t{t}"
    if kind == BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT:
        return f"bin_batched_outer_product_noncompact_torchlike_output_j{j}_k{k}_o{o}_t{t}"
    if kind == BATCHED_OUTER_NONCOMPACT_LHS_SCALAR:
        return f"bin_batched_outer_product_noncompact_lhs_scalar_j{j}_k{k}_o{o}_t{t}"
    if kind == BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP:
        return f"bin_batched_outer_product_noncompact_single_outer_group_j{j}_k{k}_o{o}_t{t}"
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
        # CPU eager ops are synchronous; this prevents accidental dead-code-like
        # benchmark changes if the body is later refactored.
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


def make_elementwise_case(torch, dtype: str, n: int):
    lhs = randn_tensor(torch, (n, n), dtype)
    rhs = randn_tensor(torch, (n, n), dtype)
    out = empty_tensor(torch, (n, n), dtype)
    return f"{n}x{n}", lambda: torch.mul(lhs, rhs, out=out)


def make_outer_product_case(torch, dtype: str, n: int):
    lhs = randn_tensor(torch, (n, 1), dtype)
    rhs = randn_tensor(torch, (1, n), dtype)
    out = empty_tensor(torch, (n, n), dtype)
    return f"{n}x{n}", lambda: torch.mul(lhs, rhs, out=out)


def make_batched_outer_compact_case(torch, dtype: str, j: int, k: int, o: int, t: int):
    lhs = randn_tensor(torch, (t, k, j), dtype)
    rhs = randn_tensor(torch, (t, o), dtype)
    out = empty_tensor(torch, (t, o, k, j), dtype)
    lhs_view = lhs[:, None, :, :]
    rhs_view = rhs[:, :, None, None]
    return f"j={j};k={k};o={o};t={t}", lambda: torch.mul(lhs_view, rhs_view, out=out)


def make_batched_outer_noncompact_case(torch, dtype: str, j: int, k: int, o: int, t: int):
    lhs = randn_tensor(torch, (t, j, k), dtype)
    rhs = randn_tensor(torch, (t, o), dtype)
    out = empty_tensor(torch, (t, o, k, j), dtype)
    lhs_view = lhs[:, None, :, :].permute(0, 1, 3, 2)
    rhs_view = rhs[:, :, None, None]
    return f"j={j};k={k};o={o};t={t}", lambda: torch.mul(lhs_view, rhs_view, out=out)


def make_batched_outer_noncompact_torchlike_output_case(torch, dtype: str, j: int, k: int, o: int, t: int):
    lhs = randn_tensor(torch, (t, j, k), dtype)
    rhs = randn_tensor(torch, (t, o), dtype)
    out = torch.empty_strided((t, o, k, j), (o * k * j, k * j, 1, k), dtype=torch_dtype(torch, dtype))
    lhs_view = lhs[:, None, :, :].permute(0, 1, 3, 2)
    rhs_view = rhs[:, :, None, None]
    return f"j={j};k={k};o={o};t={t}", lambda: torch.mul(lhs_view, rhs_view, out=out)


def make_batched_outer_noncompact_lhs_scalar_case(torch, dtype: str, j: int, k: int, o: int, t: int):
    lhs = randn_tensor(torch, (t, o), dtype)
    rhs = randn_tensor(torch, (t, j, k), dtype)
    out = empty_tensor(torch, (t, o, k, j), dtype)
    lhs_view = lhs[:, :, None, None]
    rhs_view = rhs[:, None, :, :].permute(0, 1, 3, 2)
    return f"j={j};k={k};o={o};t={t}", lambda: torch.mul(lhs_view, rhs_view, out=out)


def permuted_elementwise_axes(rank: int) -> tuple[int, ...]:
    if rank == 16:
        return (7, 4, 10, 5, 12, 2, 9, 13, 1, 3, 6, 15, 14, 11, 8, 0)
    return tuple(reversed(range(rank)))


def make_permuted_elementwise_case(torch, dtype: str, rank: int, extent: int):
    shape = (extent,) * rank
    lhs = randn_tensor(torch, shape, dtype)
    rhs_base = randn_tensor(torch, shape, dtype)
    rhs = rhs_base.permute(permuted_elementwise_axes(rank))
    out = empty_tensor(torch, shape, dtype)
    return f"rank={rank};extent={extent}", lambda: torch.mul(lhs, rhs, out=out)


def make_case(torch, dtype: str, kind: str, dims: tuple[int, ...]):
    if kind == ELEMENTWISE:
        return make_elementwise_case(torch, dtype, dims[0])
    if kind == OUTER_PRODUCT:
        return make_outer_product_case(torch, dtype, dims[0])
    if kind == BATCHED_OUTER_COMPACT:
        return make_batched_outer_compact_case(torch, dtype, *dims)
    if kind == BATCHED_OUTER_NONCOMPACT:
        return make_batched_outer_noncompact_case(torch, dtype, *dims)
    if kind == BATCHED_OUTER_NONCOMPACT_TORCHLIKE_OUTPUT:
        return make_batched_outer_noncompact_torchlike_output_case(torch, dtype, *dims)
    if kind == BATCHED_OUTER_NONCOMPACT_LHS_SCALAR:
        return make_batched_outer_noncompact_lhs_scalar_case(torch, dtype, *dims)
    if kind == BATCHED_OUTER_NONCOMPACT_SINGLE_OUTER_GROUP:
        return make_batched_outer_noncompact_case(torch, dtype, *dims)
    if kind == PERMUTED_ELEMENTWISE:
        return make_permuted_elementwise_case(torch, dtype, *dims)
    raise ValueError(f"unknown benchmark kind: {kind}")


def shape_label(kind: str, dims: tuple[int, ...]) -> str:
    if kind in {ELEMENTWISE, OUTER_PRODUCT}:
        return f"{dims[0]}x{dims[0]}"
    if kind == PERMUTED_ELEMENTWISE:
        rank, extent = dims
        return f"rank={rank};extent={extent}"
    j, k, o, t = dims
    return f"j={j};k={k};o={o};t={t}"


def emit_error_rows(writer: csv.DictWriter, args: argparse.Namespace, status: str) -> None:
    for dtype in parse_dtypes(args.dtypes):
        for kind, dims in case_specs(args.profile):
            writer.writerow(
                {
                    "suite": "mul",
                    "benchmark": benchmark_name(kind, dims),
                    "dtype": dtype,
                    "threads": args.num_threads,
                    "shape": shape_label(kind, dims),
                    "backend": "pytorch-cpu",
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
                shape, fn = make_case(torch, dtype, kind, dims)
                median_ms, iqr_ms = bench(fn, args.runs, args.warmups)
                status = "ok"
                median = f"{median_ms:.6f}"
                iqr = f"{iqr_ms:.6f}"
            except Exception as exc:
                shape = ""
                status = f"error: {exc}"
                median = ""
                iqr = ""

            writer.writerow(
                {
                    "suite": "mul",
                    "benchmark": benchmark_name(kind, dims),
                    "dtype": dtype,
                    "threads": args.num_threads,
                    "shape": shape,
                    "backend": "pytorch-cpu",
                    "median_ms": median,
                    "iqr_ms": iqr,
                    "status": status,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-threads", type=positive_int, default=int(os.environ.get("OMP_NUM_THREADS", "1")))
    parser.add_argument("--profile", choices=["smoke", "quick", "full"], default=os.environ.get("STRIDED_KERNEL_MUL_BENCH_PROFILE", "full"))
    parser.add_argument("--runs", type=positive_int, default=int(os.environ.get("STRIDED_KERNEL_MUL_BENCH_RUNS", DEFAULT_RUNS)))
    parser.add_argument("--warmups", type=int, default=int(os.environ.get("STRIDED_KERNEL_MUL_BENCH_WARMUPS", DEFAULT_WARMUPS)))
    parser.add_argument("--dtypes", default=os.environ.get("STRIDED_KERNEL_MUL_BENCH_DTYPES", "f64"))
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
