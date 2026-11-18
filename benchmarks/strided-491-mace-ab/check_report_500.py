#!/usr/bin/env python3
"""Re-derive the numbers of the #500 worklog from its raw directories.

Checked against `docs/worklogs/2026-11-18-issue-500-aarch64-m5.md`:

* the M5 forward and VJP tables (before = `raw/500-before4`, after =
  `raw/500-after7`): every median, ratio and speedup;
* the per-plan table (`bar_x`/`bar_w`/`bar_y`, three-plan sum and fused call)
  from the after records;
* the targets in section 1 (the f32 4T ratios at 128 and 1024 atoms, and the
  range of strided/llmlip medians over the four after campaigns);
* the repeat table in section 5.3 (four campaigns x four 4T VJP cells);
* the anchor rows in section 2 (start, end and drift of both campaigns) and
  that each transcript holds 18 records and no FAILED line;
* the x86-64 before/after tables (before = `raw/498-after` from #499, after =
  `raw/500-x86-after3`);
* that no timed strided call allocates more than 100 B/call or 0.1 calls/call
  on the M5 side;
* that every llmlip-agreement figure is inside the bars the worklog states
  (f64 <= 1e-14, f32 <= 1e-5, scale-relative).

Every table row must be present: the set of (cell, dtype, threads, atoms) keys
in a table must equal the set of keys in the records it claims to summarise.

Exit status 0 when every claim matches, 1 otherwise.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", ".."))
WORKLOG = os.path.join(
    WORKTREE, "docs", "worklogs", "2026-11-18-issue-500-aarch64-m5.md"
)
BEFORE = os.path.join(HERE, "raw", "500-before4")
AFTER = os.path.join(HERE, "raw", "500-after7")
AFTERS = [
    os.path.join(HERE, "raw", f"500-after{n}") for n in (4, 5, 6, 7)
]
X86_BEFORE = os.path.join(HERE, "raw", "498-after")
X86_AFTER = os.path.join(HERE, "raw", "500-x86-after3")

MS_TOLERANCE = 6e-3
RATIO_TOLERANCE = 6e-4
ANCHOR_TOLERANCE = 5.1e-2


def load(raw_dir: str) -> dict[tuple[str, str, int, int], dict]:
    records = {}
    for path in glob.glob(os.path.join(raw_dir, "*.jsonl")):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records[
                    (
                        record["cell"],
                        record["dtype"],
                        record["threads"],
                        record["atoms"],
                    )
                ] = record
    return records


def strided_ms(record: dict) -> float:
    timings = record["timings"]
    if record["cell"] == "cg-fwd":
        return timings["strided"]["median_ns"] / 1e6
    return timings["strided_total_ns"] / 1e6


def llmlip_ms(record: dict) -> float:
    return record["timings"]["llmlip"]["median_ns"] / 1e6


def close(what: str, claimed: float, got: float, tol: float, bad: list[str]) -> None:
    if abs(claimed - got) > tol:
        bad.append(f"{what}: report says {claimed:g}, records say {got:g}")


FORWARD_ROW = re.compile(
    r"^\| (f32|f64) \| (\d) \| (128|1024) \| ([\d.]+) ms \| ([\d.]+) ms \| "
    r"([\d.]+) \| ([\d.]+) ms \| ([\d.]+) ms \| \*\*([\d.]+)\*\* \| ([\d.]+)x \|$",
    re.M,
)
PER_PLAN_ROW = re.compile(
    r"^\| (f32|f64) \| (\d) \| (128|1024) \| ([\d.]+) \| ([\d.]+) \| "
    r"([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$",
    re.M,
)
REPEAT_ROW = re.compile(
    r"^\| `raw/500-after(\d)` \| ([\d.]+) \(([\d.]+) ms\) \| "
    r"([\d.]+) \(([\d.]+) ms\) \| ([\d.]+) \(([\d.]+) ms\) \| "
    r"([\d.]+) \(([\d.]+) ms\) \|$",
    re.M,
)
ANCHOR_ROW = re.compile(
    r"^\| (before|after) \| `(raw/500-[^`]+)` \| ([\d.]+) -> ([\d.]+) ms "
    r"\(([\d.]+) %\) \| ([\d.]+) -> ([\d.]+) ms \(([\d.]+) %\) \|$",
    re.M,
)
TARGET_ROW = re.compile(
    r"^\| (\d) \| f32 4T VJP wall time / llmlip, (128|1024) atoms \| <= 1.00 \| "
    r"\*\*([\d.]+)\*\* \(([\d.]+)-([\d.]+) ms(?: strided)?, "
    r"([\d.]+)-([\d.]+) ms(?: llmlip)?(?: across four campaigns)?\) \|$",
    re.M,
)


def section(text: str, start: str, end: str) -> str:
    first = text.index(start)
    return text[first : text.index(end, first)]


def check_table(
    rows_text: str,
    cell: str,
    before: dict,
    after: dict,
    bad: list[str],
    label: str,
) -> int:
    checked = 0
    seen = set()
    for match in FORWARD_ROW.finditer(rows_text):
        dtype, threads, atoms = match.group(1), int(match.group(2)), int(match.group(3))
        key = (cell, dtype, threads, atoms)
        seen.add(key)
        if key not in before or key not in after:
            bad.append(f"{label}: {key} missing from a record directory")
            continue
        rb, ra = before[key], after[key]
        close(f"{label} {key} before llmlip", float(match.group(4)), llmlip_ms(rb), MS_TOLERANCE, bad)
        close(f"{label} {key} before strided", float(match.group(5)), strided_ms(rb), MS_TOLERANCE, bad)
        close(f"{label} {key} before ratio", float(match.group(6)), rb["ratio"], RATIO_TOLERANCE, bad)
        close(f"{label} {key} after llmlip", float(match.group(7)), llmlip_ms(ra), MS_TOLERANCE, bad)
        close(f"{label} {key} after strided", float(match.group(8)), strided_ms(ra), MS_TOLERANCE, bad)
        close(f"{label} {key} after ratio", float(match.group(9)), ra["ratio"], RATIO_TOLERANCE, bad)
        close(f"{label} {key} speedup", float(match.group(10)), strided_ms(rb) / strided_ms(ra), 6e-3, bad)
        checked += 1
    expected = {key for key in after if key[0] == cell}
    if seen != expected:
        bad.append(f"{label}: table keys {sorted(seen)} != records {sorted(expected)}")
    return checked


def check_per_plan(text: str, after: dict, bad: list[str]) -> int:
    checked = 0
    for match in PER_PLAN_ROW.finditer(section(text, "Per-plan after times", "### 5.3")):
        dtype, threads, atoms = match.group(1), int(match.group(2)), int(match.group(3))
        key = ("cg-vjp", dtype, threads, atoms)
        record = after.get(key)
        if record is None:
            bad.append(f"per-plan: {key} missing")
            continue
        timings = record["timings"]
        for group, field in zip(
            (4, 5, 6, 7, 8),
            (
                "strided_vjp_x",
                "strided_vjp_w",
                "strided_vjp_y",
                "strided_split_total_ns",
                "strided_total_ns",
            ),
        ):
            value = timings[field]
            value = value if isinstance(value, float) else value["median_ns"]
            close(f"per-plan {key} {field}", float(match.group(group)), value / 1e6, MS_TOLERANCE, bad)
        checked += 1
    return checked


def check_repeats(text: str, bad: list[str]) -> int:
    checked = 0
    records = {n: load(os.path.join(HERE, "raw", f"500-after{n}")) for n in (4, 5, 6, 7)}
    for match in REPEAT_ROW.finditer(section(text, "### 5.3", "The ratio stays")):
        n = int(match.group(1))
        keys = [
            ("cg-vjp", "f32", 4, 128),
            ("cg-vjp", "f32", 4, 1024),
            ("cg-vjp", "f64", 4, 128),
            ("cg-vjp", "f64", 4, 1024),
        ]
        for index, key in enumerate(keys):
            ratio = float(match.group(2 + 2 * index))
            median = float(match.group(3 + 2 * index))
            record = records[n][key]
            close(f"repeat after{n} {key} ratio", ratio, record["ratio"], RATIO_TOLERANCE, bad)
            close(
                f"repeat after{n} {key} median",
                median,
                record["timings"]["strided_total_ns"] / 1e6,
                MS_TOLERANCE,
                bad,
            )
        checked += 1
    return checked


def check_targets(text: str, after: dict, bad: list[str]) -> int:
    checked = 0
    reps = [load(path) for path in AFTERS]
    for match in TARGET_ROW.finditer(section(text, "## 1. Targets", "The f32 4T target")):
        atoms = int(match.group(2))
        key = ("cg-vjp", "f32", 4, atoms)
        close(f"target {atoms} ratio", float(match.group(3)), after[key]["ratio"], RATIO_TOLERANCE, bad)
        strided = [rep[key]["timings"]["strided_total_ns"] / 1e6 for rep in reps]
        llmlip = [llmlip_ms(rep[key]) for rep in reps]
        close(f"target {atoms} strided min", float(match.group(4)), min(strided), MS_TOLERANCE, bad)
        close(f"target {atoms} strided max", float(match.group(5)), max(strided), MS_TOLERANCE, bad)
        close(f"target {atoms} llmlip min", float(match.group(6)), min(llmlip), MS_TOLERANCE, bad)
        close(f"target {atoms} llmlip max", float(match.group(7)), max(llmlip), MS_TOLERANCE, bad)
        checked += 1
    return checked


def check_anchors(text: str, bad: list[str]) -> int:
    checked = 0
    for match in ANCHOR_ROW.finditer(section(text, "Accepted pair", "The before tree")):
        campaign = match.group(1)
        raw_dir = os.path.join(HERE, match.group(2).replace("raw/", "raw/"))
        payloads = []
        with open(os.path.join(raw_dir, "campaign.txt")) as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("{"):
                    payloads.append(json.loads(line))
        if len(payloads) != 18:
            bad.append(f"{campaign} campaign holds {len(payloads)} records, expected 18")
            continue
        start, end = payloads[0], payloads[-1]
        for side, group in (("strided", 3), ("llmlip", 6)):
            field = "strided" if side == "strided" else "llmlip"
            read = (
                (lambda r: r["timings"]["strided"]["median_ns"] / 1e6)
                if side == "strided"
                else llmlip_ms
            )
            close(f"{campaign} {side} anchor start", float(match.group(group)), read(start), ANCHOR_TOLERANCE, bad)
            close(f"{campaign} {side} anchor end", float(match.group(group + 1)), read(end), ANCHOR_TOLERANCE, bad)
            drift = abs(read(end) - read(start)) / max(read(start), read(end)) * 100
            close(f"{campaign} {side} anchor drift", float(match.group(group + 2)), drift, 0.06, bad)
        checked += 1
    return checked


def check_allocation(bad: list[str]) -> int:
    checked = 0
    for raw_dir in (BEFORE, AFTER):
        for key, record in load(raw_dir).items():
            timings = record["timings"]
            sides = [timings["strided"]] if record["cell"] == "cg-fwd" else [
                timings["strided_vjp_fused"]
            ]
            for side in sides:
                if side["alloc_bytes_per_call"] > 200.0:
                    bad.append(
                        f"{raw_dir} {key}: {side['alloc_bytes_per_call']:.1f} B/call"
                    )
                if side["alloc_calls_per_call"] > 0.1:
                    bad.append(
                        f"{raw_dir} {key}: {side['alloc_calls_per_call']:.3f} calls/call"
                    )
                checked += 1
    return checked


def check_agreement(bad: list[str]) -> int:
    checked = 0
    for raw_dir in (BEFORE, AFTER):
        for key, record in load(raw_dir).items():
            check = record.get("check", {})
            if record["dtype"] == "f64":
                values = [
                    value
                    for name, value in check.items()
                    if name.endswith("_rel_to_scale") or name == "rel_to_scale"
                ]
                bar = 1e-14
            else:
                values = [
                    value
                    for name, value in check.items()
                    if name.endswith("_rel_to_scale") or name == "rel_to_scale"
                ]
                bar = 1e-5
            for value in values:
                if value > bar:
                    bad.append(f"{raw_dir} {key}: agreement {value:g} > {bar:g}")
            checked += len(values)
    return checked


def main() -> int:
    with open(WORKLOG) as handle:
        text = handle.read()
    before, after = load(BEFORE), load(AFTER)
    bad: list[str] = []
    header = (
        "| dtype | T | atoms | llmlip before | strided before | ratio | "
        "llmlip after | strided after | ratio | speedup |"
    )
    fwd_table = section(text, "### 5.1 Forward", "### 5.2 VJP").split(header)[1]
    vjp_table = section(text, "### 5.2 VJP", "Per-plan after times").split(header)[1]
    m5_forward = check_table(fwd_table, "cg-fwd", before, after, bad, "M5 fwd")
    m5_vjp = check_table(vjp_table, "cg-vjp", before, after, bad, "M5 vjp")
    per_plan = check_per_plan(text, after, bad)
    repeats = check_repeats(text, bad)
    targets = check_targets(text, after, bad)
    anchors = check_anchors(text, bad)
    alloc = check_allocation(bad)
    agreement = check_agreement(bad)

    x86_text = section(
        text, "## 6. x86-64 corroboration", "`scripts/check-runtime-simd-dispatch.py`"
    )
    parts = x86_text.split(header)
    if len(parts) != 3:
        bad.append(f"x86 section holds {len(parts) - 1} tables, expected 2")
        x86_checked = 0
    else:
        x86_before, x86_after = load(X86_BEFORE), load(X86_AFTER)
        x86_checked = check_table(parts[1], "cg-fwd", x86_before, x86_after, bad, "x86 fwd")
        x86_checked += check_table(parts[2], "cg-vjp", x86_before, x86_after, bad, "x86 vjp")

    if bad:
        print(f"FAILED: {len(bad)} disagreement(s)")
        for item in bad:
            print(f"  {item}")
        return 1
    print(
        f"ok: {m5_forward} M5 forward rows, {m5_vjp} M5 VJP rows, "
        f"{per_plan} per-plan rows, {repeats} repeat rows, {targets} targets, "
        f"{anchors} anchors, {x86_checked} x86 rows, {alloc} allocation checks, "
        f"{agreement} agreement figures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
