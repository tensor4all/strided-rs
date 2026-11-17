#!/usr/bin/env python3
"""Check every number the #496 report states against the raw records.

The report (docs/worklogs/2026-11-17-issue-496-segpoly-simd-reduce.md) has seven
tables plus a handful of inline numbers, and the raw records are
`raw/496-before/cg-*.jsonl`, `raw/496-after/cg-*.jsonl` and
`raw/496-scaling/cg-*.jsonl`. This script re-derives the report's claims from the
records and exits nonzero on any disagreement, so "the tables match the records"
is a command, not a promise.

Usage: ./check_report_496.py [--worklog PATH] [--before DIR] [--after DIR]
                             [--scaling DIR]
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKLOG = os.path.join(
    HERE,
    "..",
    "..",
    "docs",
    "worklogs",
    "2026-11-17-issue-496-segpoly-simd-reduce.md",
)
MS = 1e6


def load_records(raw_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "cg-*.jsonl"))):
        rec = json.loads(open(path).read().strip())
        records[(rec["cell"], rec["dtype"], rec["threads"], rec["atoms"])] = rec
    return records


def timing(rec, side):
    return rec["timings"][side]["median_ns"] / MS


def vjp_sum(rec):
    """The three plans' sum, the number comparable with llmlip's fused pass."""
    return sum(timing(rec, f"strided_vjp_{which}") for which in "xwy")


def cell_ms(rec):
    """One cell's strided wall time, summed over plans for the VJP cells."""
    if rec["cell"] == "cg-fwd":
        return timing(rec, "strided")
    return vjp_sum(rec)


class Report:
    def __init__(self):
        self.bad = []
        self.checks = 0

    def near(self, what, got, claim, tol):
        self.checks += 1
        if abs(got - claim) > tol:
            self.bad.append(f"{what}: records say {got:.4g}, report says {claim} (tol {tol})")

    def same(self, what, got, claim):
        self.checks += 1
        if got != claim:
            self.bad.append(f"{what}: records say {got!r}, report says {claim!r}")


def check_forward(text, before, after, report):
    """§4: before llmlip / strided / ratio, after the same, and the speedup."""
    row = re.compile(
        r"^\| (f32|f64) \| (\d) \| (\d+) \| ([\d.]+) ms \| ([\d.]+) ms \| \*{0,2}([\d.]+)\*{0,2} \| "
        r"([\d.]+) ms \| ([\d.]+) ms \| \*{0,2}([\d.]+)\*{0,2} \| ([\d.]+)x \|$",
        re.M,
    )
    for m in row.finditer(text):
        dtype, threads, atoms = m.group(1), int(m.group(2)), int(m.group(3))
        key = ("cg-fwd", dtype, threads, atoms)
        old, new = before[key], after[key]
        report.near(
            f"fwd {dtype} {threads}T {atoms} before llmlip", timing(old, "llmlip"), float(m.group(4)), 0.051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} before strided", timing(old, "strided"), float(m.group(5)), 0.051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} before ratio", old["ratio"], float(m.group(6)), 0.0051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} after llmlip", timing(new, "llmlip"), float(m.group(7)), 0.051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} after strided", timing(new, "strided"), float(m.group(8)), 0.051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} after ratio", new["ratio"], float(m.group(9)), 0.0051
        )
        report.near(
            f"fwd {dtype} {threads}T {atoms} speedup",
            timing(old, "strided") / timing(new, "strided"),
            float(m.group(10)),
            0.0051,
        )


def check_dtype_ratio(text, after, report):
    """§4: f32/f64 on the same cell, both sides."""
    row = re.compile(
        r"^\| (\dT \d+) \| ([\d.]+) \| ([\d.]+) \|$",
        re.M,
    )
    cells = {"1T 128": (1, 128), "1T 1024": (1, 1024), "4T 128": (4, 128), "4T 1024": (4, 1024)}
    for m in row.finditer(text):
        threads, atoms = cells[m.group(1)]
        for side, column in (("llmlip", m.group(2)), ("strided", m.group(3))):
            f32 = cell_ms(after[("cg-fwd", "f32", threads, atoms)]) if side == "strided" else timing(
                after[("cg-fwd", "f32", threads, atoms)], "llmlip"
            )
            f64 = cell_ms(after[("cg-fwd", "f64", threads, atoms)]) if side == "strided" else timing(
                after[("cg-fwd", "f64", threads, atoms)], "llmlip"
            )
            report.near(f"f32/f64 {m.group(1)} {side}", f32 / f64, float(column), 0.0051)


def check_vjp(text, before, after, report):
    """§5: bar_y and the three-plan sum, before and after, with the speedup."""
    row = re.compile(
        r"^\| (f32|f64) \| (\d) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| "
        r"([\d.]+) \| ([\d.]+) \| ([\d.]+)x \|$",
        re.M,
    )
    for m in row.finditer(text):
        dtype, threads, atoms = m.group(1), int(m.group(2)), int(m.group(3))
        key = ("cg-vjp", dtype, threads, atoms)
        old, new = before[key], after[key]
        report.near(f"vjp {dtype} {threads}T {atoms} bar_y before", timing(old, "strided_vjp_y"), float(m.group(4)), 0.051)
        report.near(f"vjp {dtype} {threads}T {atoms} sum before", vjp_sum(old), float(m.group(5)), 0.051)
        report.near(
            f"vjp {dtype} {threads}T {atoms} before ratio",
            vjp_sum(old) / timing(old, "llmlip"),
            float(m.group(6)),
            0.0051,
        )
        report.near(f"vjp {dtype} {threads}T {atoms} bar_y after", timing(new, "strided_vjp_y"), float(m.group(7)), 0.051)
        report.near(f"vjp {dtype} {threads}T {atoms} sum after", vjp_sum(new), float(m.group(8)), 0.051)
        report.near(
            f"vjp {dtype} {threads}T {atoms} after ratio",
            vjp_sum(new) / timing(new, "llmlip"),
            float(m.group(9)),
            0.0051,
        )
        report.near(
            f"vjp {dtype} {threads}T {atoms} bar_y speedup",
            timing(old, "strided_vjp_y") / timing(new, "strided_vjp_y"),
            float(m.group(10)),
            0.051,
        )


def check_scaling(text, after, report):
    """§7: 1T/4T per cell for both sides."""
    row = re.compile(
        r"^\| (cg-fwd|cg-vjp) \| (f32|f64) \| (\d+) \| ([\d.]+) ms \| ([\d.]+) ms \| "
        r"\*{0,2}([\d.]+)x\*{0,2} \| ([\d.]+)x \|$",
        re.M,
    )
    for m in row.finditer(text):
        cell, dtype, atoms = m.group(1), m.group(2), int(m.group(3))
        one, four = after[(cell, dtype, 1, atoms)], after[(cell, dtype, 4, atoms)]
        report.near(f"scaling {cell} {dtype} {atoms} 1T", cell_ms(one), float(m.group(4)), 0.051)
        report.near(f"scaling {cell} {dtype} {atoms} 4T", cell_ms(four), float(m.group(5)), 0.051)
        report.near(
            f"scaling {cell} {dtype} {atoms} strided ratio",
            cell_ms(one) / cell_ms(four),
            float(m.group(6)),
            0.0051,
        )
        report.near(
            f"scaling {cell} {dtype} {atoms} llmlip ratio",
            timing(one, "llmlip") / timing(four, "llmlip"),
            float(m.group(7)),
            0.0051,
        )


def check_sweep(text, scaling, report):
    """§7: the 1/2/3/4T sweep of one cell, its scaling and per-thread efficiency."""
    row = re.compile(
        r"^\| \d \((\d(?:,\d)*)\) \| ([\d.]+) ms \| ([\d.]+)x \| ([\d.]+) \| "
        r"([\d.]+) ms \| ([\d.]+)x \|$",
        re.M,
    )
    one = scaling[("cg-fwd", "f64", 1, 1024)]
    for m in row.finditer(text):
        threads = len(m.group(1).split(","))
        rec = scaling[("cg-fwd", "f64", threads, 1024)]
        report.near(f"sweep {threads}T strided", timing(rec, "strided"), float(m.group(2)), 0.051)
        scaling_factor = timing(one, "strided") / timing(rec, "strided")
        report.near(f"sweep {threads}T scaling", scaling_factor, float(m.group(3)), 0.0051)
        report.near(
            f"sweep {threads}T efficiency", scaling_factor / threads, float(m.group(4)), 0.0051
        )
        report.near(f"sweep {threads}T llmlip", timing(rec, "llmlip"), float(m.group(5)), 0.051)
        report.near(
            f"sweep {threads}T llmlip scaling",
            timing(one, "llmlip") / timing(rec, "llmlip"),
            float(m.group(6)),
            0.051,
        )


def check_absolute_4t(flat, after, report):
    """§7's closing sentence: the four 4T wall times, both sides."""
    claim = re.compile(
        r"(\d+\.\d) vs (\d+\.\d) ms \(f64 1024\), ([\d.]+) vs ([\d.]+) \(f32 1024\), "
        r"([\d.]+) vs ([\d.]+) \(f64 128\), ([\d.]+) vs ([\d.]+) \(f32 128\)"
    )
    found = claim.search(flat)
    if not found:
        report.bad.append("§7: the four 4T absolute pairs are missing")
        return
    wanted = [
        ("f64", 1024),
        ("f32", 1024),
        ("f64", 128),
        ("f32", 128),
    ]
    groups = found.groups()
    for index, (dtype, atoms) in enumerate(wanted):
        rec = after[("cg-fwd", dtype, 4, atoms)]
        report.near(f"4T {dtype} {atoms} strided", timing(rec, "strided"), float(groups[2 * index]), 0.051)
        report.near(f"4T {dtype} {atoms} llmlip", timing(rec, "llmlip"), float(groups[2 * index + 1]), 0.051)


def check_allocations(text, before, after, report):
    """§6: 1T cells are 0/0; a 4T event is the 1 520-byte rayon thread start-up."""
    for name, records in (("before", before), ("after", after)):
        for key, rec in records.items():
            for side, numbers in rec["timings"].items():
                if not isinstance(numbers, dict) or not side.startswith("strided"):
                    continue
                report.checks += 1
                bytes_total = numbers["alloc_bytes_total"]
                calls_total = numbers["alloc_calls_total"]
                if key[2] == 1:
                    if bytes_total or calls_total:
                        report.bad.append(
                            f"{name} {key} {side}: a 1T cell allocates {bytes_total} B "
                            f"in {calls_total} calls"
                        )
                    continue
                if not calls_total:
                    continue
                if bytes_total != 1520 * calls_total:
                    report.bad.append(
                        f"{name} {key} {side}: {calls_total} events of {bytes_total} B, "
                        f"not 1 520 bytes each"
                    )
                if calls_total * 20 > numbers["timed_calls"]:
                    report.bad.append(
                        f"{name} {key} {side}: {calls_total} events in "
                        f"{numbers['timed_calls']} calls is above 5 % -- more than a "
                        f"start-up event"
                    )
    # The two llmlip figures §6 quotes, as bytes and calls per call.
    seen = {
        (int(numbers["alloc_bytes_per_call"]), int(numbers["alloc_calls_per_call"]))
        for rec in list(before.values()) + list(after.values())
        for side, numbers in rec["timings"].items()
        if isinstance(numbers, dict)
    }
    for bytes_text, calls_text in re.findall(r"(\d[\d ]*\d|\d) B in ([\d ]+) calls", text):
        want = (int(bytes_text.replace(" ", "")), int(calls_text.replace(" ", "")))
        report.same(f"llmlip allocation quote {want[0]} B", want in seen, True)


def check_numerics(flat, before, after, report):
    """§8: the forward and bar_x/bar_w checks are unchanged, bar_y is quoted."""
    for key, old in before.items():
        new = after[key]
        if key[0] == "cg-fwd":
            report.same(f"fwd {key[1:]} rel_to_scale", new["check"]["rel_to_scale"], old["check"]["rel_to_scale"])
        else:
            for which in ("bar_x_rel_to_scale", "bar_w_rel_to_scale"):
                report.same(f"{key[1:]} {which}", new["check"][which], old["check"][which])
    # The four bar_y pairs §8 quotes, as (dtype, atoms, before, after) in e-15/e-6.
    pairs = re.findall(r"(f32|f64) (\d+) atoms ([\d.]+)e-(\d+) (?:->|→) ([\d.]+)e-(\d+)", flat)
    for dtype, atoms, old_mantissa, old_exp, new_mantissa, new_exp in pairs:
        threads = 1
        key = ("cg-vjp", dtype, threads, int(atoms))
        report.near(
            f"bar_y_rel_to_scale {dtype} {atoms} before",
            before[key]["check"]["bar_y_rel_to_scale"],
            float(f"{old_mantissa}e-{old_exp}"),
            5e-3 * float(f"{old_mantissa}e-{old_exp}"),
        )
        report.near(
            f"bar_y_rel_to_scale {dtype} {atoms} after",
            after[key]["check"]["bar_y_rel_to_scale"],
            float(f"{new_mantissa}e-{new_exp}"),
            5e-3 * float(f"{new_mantissa}e-{new_exp}"),
        )


def check_rss(flat, before, after, report):
    """§6: the two peak-RSS pairs it quotes explicitly, to 0.1 MiB."""
    keys = {
        "fwd f64 1T 128": ("cg-fwd", "f64", 1, 128),
        "VJP f64 1T 1024": ("cg-vjp", "f64", 1, 1024),
    }
    for m in re.finditer(r"(fwd f64 1T 128|VJP f64 1T 1024) ([\d.]+) → ([\d.]+) MiB", flat):
        key = keys[m.group(1)]
        report.near(
            f"rss {m.group(1)} before", before[key]["rss_bytes"] / 2**20, float(m.group(2)), 0.11
        )
        report.near(
            f"rss {m.group(1)} after", after[key]["rss_bytes"] / 2**20, float(m.group(3)), 0.11
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worklog", default=DEFAULT_WORKLOG)
    parser.add_argument("--before", default=os.path.join(HERE, "raw", "496-before"))
    parser.add_argument("--after", default=os.path.join(HERE, "raw", "496-after"))
    parser.add_argument("--scaling", default=os.path.join(HERE, "raw", "496-scaling"))
    args = parser.parse_args()

    # Table rows are matched with `^|` anchors, prose claims after wrapping: keep
    # one line per line for the tables and a single line for the prose.
    raw_text = open(args.worklog).read()
    text = "\n".join(" ".join(line.split()) for line in raw_text.splitlines())
    flat = " ".join(raw_text.split())
    before = load_records(args.before)
    after = load_records(args.after)
    scaling = load_records(args.scaling)
    if not before or not after or not scaling:
        print(f"no records: {args.before}, {args.after}, {args.scaling}")
        return 1

    report = Report()
    check_forward(text, before, after, report)
    check_dtype_ratio(text, after, report)
    check_vjp(text, before, after, report)
    check_scaling(text, after, report)
    check_sweep(text, scaling, report)
    check_absolute_4t(flat, after, report)
    check_allocations(flat, before, after, report)
    check_numerics(flat, before, after, report)
    check_rss(flat, before, after, report)
    for line in report.bad:
        print(f"FAIL {line}")
    if report.bad:
        return 1
    print(f"ok: {report.checks} claims agree with the records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
