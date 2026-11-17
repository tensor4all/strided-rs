#!/usr/bin/env python3
"""Check every number the report states against the raw records.

The report (docs/worklogs/2026-11-17-issue-491-segpoly-vs-llmlip.md) is prose plus
four tables, and the raw records are raw/cg-*.jsonl. This script re-derives the
report's claims from the records and exits nonzero on any disagreement, so
"the tables match the records" is a command, not a promise.

Usage: ./check_report.py [--worklog PATH] [--raw-dir PATH]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKLOG = os.path.join(
    HERE, "..", "..", "docs", "worklogs", "2026-11-17-issue-491-segpoly-vs-llmlip.md"
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


def check_tables(text, records, bad):
    """§4 forward table and §5 VJP table, row by row."""
    fwd = re.compile(
        r"^\| (f32|f64) \| (\d) \| (\d+) \| ([\d.]+) ms \| ([\d.]+) ms \| \*{0,2}([\d.]+)\*{0,2} \|$",
        re.M,
    )
    n_fwd = 0
    for m in fwd.finditer(text):
        dtype, threads, atoms = m.group(1), int(m.group(2)), int(m.group(3))
        rec = records[("cg-fwd", dtype, threads, atoms)]
        llmlip, strided = timing(rec, "llmlip"), timing(rec, "strided")
        n_fwd += 1
        for name, got, claim, tol in (
            ("llmlip ms", llmlip, float(m.group(4)), 0.051),
            ("strided ms", strided, float(m.group(5)), 0.051),
            ("ratio", strided / llmlip, float(m.group(6)), 0.0051),
        ):
            if abs(got - claim) > tol:
                bad.append(f"fwd {dtype} {threads}T {atoms}: {name} is {got:.3f}, report says {claim}")

    vjp = re.compile(
        r"^\| (f32|f64) \| (\d) \| (\d+) \| ([\d.]+) ms \| ([\d.]+) \(([\d.]+)\) \| "
        r"([\d.]+) \(([\d.]+)\) \| ([\d.]+) \(([\d.]+)\) \| ([\d.]+) \| ([\d.]+) \|$",
        re.M,
    )
    n_vjp = 0
    for m in vjp.finditer(text):
        g = m.groups()
        dtype, threads, atoms = g[0], int(g[1]), int(g[2])
        rec = records[("cg-vjp", dtype, threads, atoms)]
        llmlip = timing(rec, "llmlip")
        x, w, y = (timing(rec, s) for s in ("strided_vjp_x", "strided_vjp_w", "strided_vjp_y"))
        total = x + w + y
        n_vjp += 1
        claims = (
            ("llmlip ms", llmlip, float(g[3]), 0.051),
            ("bar_x ms", x, float(g[4]), 0.051),
            ("bar_x ratio", x / llmlip, float(g[5]), 0.0051),
            ("bar_w ms", w, float(g[6]), 0.051),
            ("bar_w ratio", w / llmlip, float(g[7]), 0.0051),
            ("bar_y ms", y, float(g[8]), 0.051),
            ("bar_y ratio", y / llmlip, float(g[9]), 0.0051),
            ("total ms", total, float(g[10]), 0.051),
            ("total ratio", total / llmlip, float(g[11]), 0.0051),
        )
        for name, got, claim, tol in claims:
            if abs(got - claim) > tol:
                bad.append(f"vjp {dtype} {threads}T {atoms}: {name} is {got:.3f}, report says {claim}")
    return n_fwd, n_vjp


def check_grad_x(records, bad):
    """`grad_x_mib` must be the real buffer: N*(128 + 3*f) elements."""
    checked = 0
    for key, rec in sorted(records.items()):
        if key[0] != "cg-vjp":
            continue
        element = 8 if key[1] == "f64" else 4
        want = rec["nodes"] * (128 + 3 * 128) * element / 2**20
        checked += 1
        if abs(rec["grad_x_mib"] - want) > 0.051:
            bad.append(f"{key}: grad_x_mib is {rec['grad_x_mib']}, buffer is {want:.2f} MiB")
    return checked


def check_numerics(text, records, bad):
    """§7: max|ours - llmlip| and the same number over the reference scale.

    The table has one column per atom count and, for the VJP rows, two values per
    column (1T / 4T), some of which coincide. Each claimed number must match some
    record value of the same (dtype, quantity, atoms) group within the precision
    the report prints (~2 significant digits), and every record value must be
    matched by some claimed number, so a row cannot quietly lose a thread.
    """
    quantity = {
        "forward": ("cg-fwd", "max_abs", "rel_to_scale"),
        "fwd": ("cg-fwd", "max_abs", "rel_to_scale"),
        "VJP `bar_x`": ("cg-vjp", "bar_x_abs", "bar_x_rel_to_scale"),
        "VJP `bar_w`": ("cg-vjp", "bar_w_abs", "bar_w_rel_to_scale"),
        "VJP `bar_y`": ("cg-vjp", "bar_y_abs", "bar_y_rel_to_scale"),
    }
    row = re.compile(
        r"^\| (f32|f64) (forward|fwd|VJP `bar_x`|VJP `bar_w`|VJP `bar_y`) \| ([^|]+?) \| "
        r"([^|]+?) \| ([^|]+?) \| ([^|]+?) \|$",
        re.M,
    )
    checked = 0
    for m in row.finditer(text):
        dtype, name, cols = m.group(1), m.group(2), m.groups()[2:]
        cell, key, rkey = quantity[name]
        for atoms, abs_text, rel_text in ((128, cols[0], cols[1]), (1024, cols[2], cols[3])):
            group = [
                (rec["check"][key], rec["check"][rkey])
                for k, rec in records.items()
                if k[0] == cell and k[1] == dtype and k[3] == atoms
            ]
            if not group:
                bad.append(f"no records for {dtype} {name} {atoms} atoms")
                continue
            for index, what in ((0, "abs"), (1, "of scale")):
                cell_text = re.sub(r"\(\dT\)", "", abs_text if index == 0 else rel_text)
                claimed = [float(x) for x in re.findall(r"[\d.]+e[-+]?\d+|[\d.]+", cell_text)]
                if not claimed:
                    bad.append(f"{dtype} {name} {atoms}: report prints no {what}")
                    continue
                raw = sorted({v[index] for v in group})
                for claim in claimed:
                    if not any(got == 0 and claim == 0 or got and abs(claim - got) / got <= 0.06 for got in raw):
                        bad.append(
                            f"{dtype} {name} {atoms} {what}: report says {claim:g}, "
                            f"records say {['%.2e' % v for v in raw]}"
                        )
                for got in raw:
                    if not any(got == 0 and c == 0 or got and abs(c - got) / got <= 0.06 for c in claimed):
                        bad.append(
                            f"{dtype} {name} {atoms} {what}: record value {got:.2e} is missing "
                            f"from the report ({claimed})"
                        )
                checked += 1
    return checked


def check_ranges(text, records, bad):
    """Prose ranges: bar_y share of the VJP segment, exec_new, f64 agreement."""
    shares = [
        100 * rec["timings"]["strided_vjp_y"]["median_ns"] / rec["timings"]["strided_total_ns"]
        for key, rec in records.items()
        if key[0] == "cg-vjp"
    ]
    m = re.search(r"\*{0,2}(\d+)-(\d+) %\*{0,2} of the", text)
    if not m:
        bad.append("report no longer states the bar_y share range")
    elif not (math.floor(min(shares)) == float(m.group(1)) and math.ceil(max(shares)) == float(m.group(2))):
        bad.append(f"bar_y share is {min(shares):.1f}-{max(shares):.1f} %, report says {m.group(1)}-{m.group(2)}")

    execs = [rec["exec_new"]["median_ns"] for rec in records.values()]
    m = re.search(r"\*{0,2}([\d.]+)-([\d.]+) ns and 0 bytes per mint", text)
    if not m:
        bad.append("report no longer states the exec_new range")
    elif not (f"{min(execs):.1f}" == m.group(1) and f"{max(execs):.1f}" == m.group(2)):
        bad.append(f"exec_new is {min(execs):.1f}-{max(execs):.1f} ns, report says {m.group(1)}-{m.group(2)}")

    rel = [
        rec["check"][k]
        for key, rec in records.items()
        if key[1] == "f64"
        for k in ("rel_to_scale", "bar_x_rel_to_scale", "bar_w_rel_to_scale", "bar_y_rel_to_scale")
        if k in rec["check"] and rec["check"][k]
    ]
    m = re.search(r"agrees to ([\d.]+e[-+]?\d+)-([\d.]+e[-+]?\d+) of scale", text)
    if not m:
        bad.append("report no longer states the f64 agreement range")
    else:
        lo, hi = float(m.group(1)), float(m.group(2))
        if not (f"{min(rel):.1e}" == m.group(1) and f"{max(rel):.1e}" == m.group(2)):
            bad.append(f"f64 rel_to_scale is {min(rel):.2e}-{max(rel):.2e}, report says {lo:g}-{hi:g}")
    return len(shares), len(execs), len(rel)


def check_campaign(raw_dir, records, bad):
    """The committed transcript must be the same 18 JSON lines as the cells."""
    path = os.path.join(raw_dir, "campaign.txt")
    text = open(path).read()
    lines = text.splitlines()
    payloads = [line for line in lines if line.startswith("{")]
    if len(payloads) != 18:
        bad.append(f"campaign.txt holds {len(payloads)} JSON lines, expected 18 (16 cells + 2 anchors)")
    if any("FAILED" in line for line in lines):
        bad.append("campaign.txt records a failure")
    if "=== done" not in text:
        bad.append("campaign.txt has no '=== done' sentinel")
    cells = {json.dumps(rec, sort_keys=True) for rec in records.values()}
    missing = cells - {json.dumps(json.loads(p), sort_keys=True) for p in payloads}
    if missing:
        bad.append(f"campaign.txt does not contain {len(missing)} of the raw records")
    return len(payloads)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worklog", default=DEFAULT_WORKLOG)
    parser.add_argument("--raw-dir", default=os.path.join(HERE, "raw"))
    args = parser.parse_args()

    text = open(args.worklog).read()
    records = load_records(args.raw_dir)
    bad = []
    if len(records) != 16:
        bad.append(f"found {len(records)} records in {args.raw_dir}, expected 16")

    n_fwd, n_vjp = check_tables(text, records, bad)
    n_grad = check_grad_x(records, bad)
    n_num = check_numerics(text, records, bad)
    for what, got, want in (("forward table rows", n_fwd, 8), ("VJP table rows", n_vjp, 8),
                            ("grad_x_mib cells", n_grad, 8), ("numerics columns", n_num, 32)):
        if got != want:
            bad.append(f"{what}: checked {got}, expected {want}")
    shares, execs, rel = check_ranges(text, records, bad)
    n_campaign = check_campaign(args.raw_dir, records, bad)

    print(f"forward table rows   : {n_fwd}/8")
    print(f"VJP table rows       : {n_vjp}/8")
    print(f"grad_x_mib cells     : {n_grad}/8")
    print(f"numerics columns     : {n_num}/32")
    print(f"campaign.txt lines   : {n_campaign}/18")
    print(f"bar_y share / exec_new / f64 agreement: {shares} / {execs} / {rel} values")
    if bad:
        print()
        for line in bad:
            print(f"MISMATCH {line}")
        return 1
    print("\nOK: every table row, size field, range and transcript line matches the records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
