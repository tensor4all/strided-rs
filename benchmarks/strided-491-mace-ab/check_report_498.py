#!/usr/bin/env python3
"""Re-derive the numbers of the #498 worklog from its two raw directories.

Checked against `docs/worklogs/2026-11-18-issue-498-segpoly-fused-vjp.md`:

* the forward table (before/after llmlip and strided medians, their ratios, the
  speedup), from `raw/498-before` and `raw/498-after`;
* the VJP table (the same plus the after campaign's three-plan sum), where the
  compared figure is one fused call after and the three-plan sum before;
* the targets in §1 (the f32 4T VJP ratios), the anchor rows of §3 (start, end and
  drift of each campaign), the allocation claim of §6 (0 bytes and 0 calls per
  fused call), its grad_x / bar_y-intermediate table, and the "peak RSS unchanged
  within 1 MiB" claim against both campaigns' records, and the worst
  llmlip-agreement figures of §7;
* that the speedup ranges quoted in §4 and §5 are the extremes of the table.

Every row of the two tables has to be present: the set of (cell, dtype, threads,
atoms) keys found in the report must equal the set the records hold.

The fused-groups-only row of the decomposition table in §5 is a single-cell probe
on an intermediate tree; it is not in `raw/` and this checker does not verify it
(the worklog says so in place).

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
DEFAULT_WORKLOG = os.path.join(
    WORKTREE, "docs", "worklogs", "2026-11-18-issue-498-segpoly-fused-vjp.md"
)
BEFORE = os.path.join(HERE, "raw", "498-before")
AFTER = os.path.join(HERE, "raw", "498-after")

# A printed ms has two decimals, so its rounding error is at most 5e-3; ratios and
# speedups print three and two.
MS_TOLERANCE = 6e-3
RATIO_TOLERANCE = 6e-4
# The anchor table prints one decimal, so its rounding error is up to 5e-2.
ANCHOR_TOLERANCE = 5.1e-2


def load(raw_dir: str) -> dict[tuple[str, str, int, int], dict]:
    records: dict[tuple[str, str, int, int], dict] = {}
    for path in glob.glob(os.path.join(raw_dir, "*.jsonl")):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = (
                    record["cell"],
                    record["dtype"],
                    record["threads"],
                    record["atoms"],
                )
                records[key] = record
    return records


def total_ns(record: dict) -> float:
    """The cell's strided figure: the forward call or the fused VJP call."""
    timings = record["timings"]
    if record["cell"] == "cg-fwd":
        return timings["strided"]["median_ns"]
    return timings["strided_total_ns"]


FORWARD_ROW = re.compile(
    r"^\| (f32|f64) \| (\d) \| (128|1024) \| ([\d.]+) ms \| ([\d.]+) ms \| "
    r"\*\*([\d.]+)\*\* \| ([\d.]+) ms \| ([\d.]+) ms \| \*\*([\d.]+)\*\* \| ([\d.]+)x \|$",
    re.M,
)
VJP_ROW = re.compile(
    r"^\| (f32|f64) \| (\d) \| (128|1024) \| ([\d.]+) ms \| ([\d.]+) ms \| "
    r"\*\*([\d.]+)\*\* \| ([\d.]+) ms \| ([\d.]+) ms \| ([\d.]+) ms \| "
    r"\*\*([\d.]+)\*\* \| ([\d.]+)x \|$",
    re.M,
)
BUFFER_ROW = re.compile(
    r"^\| (f32|f64) \| (128|1024) \| ([\d.]+) MiB \| ([\d.]+) MiB \|$", re.M
)
ANCHOR_ROW = re.compile(
    r"^\| (before|after) \| ([\d.]+) -> ([\d.]+) ms \| ([\d.]+) % \| "
    r"([\d.]+) -> ([\d.]+) ms \| ([\d.]+) % \|$",
    re.M,
)


def close(what: str, claimed: float, got: float, tolerance: float, bad: list[str]) -> None:
    if abs(claimed - got) > tolerance:
        bad.append(f"{what}: report says {claimed:g}, records say {got:g}")


def check_cells(
    text: str,
    before: dict,
    after: dict,
    bad: list[str],
    speedup_range: tuple[float, float],
) -> int:
    """The forward and VJP tables: every cell in both directions."""
    checked = 0
    speedups = []
    rows: set[tuple[str, str, int, int]] = set()
    for pattern, cell in ((FORWARD_ROW, "cg-fwd"), (VJP_ROW, "cg-vjp")):
        for match in pattern.finditer(text):
            dtype, threads, atoms = match.group(1), int(match.group(2)), int(match.group(3))
            key = (cell, dtype, threads, atoms)
            rows.add(key)
            if key not in before or key not in after:
                bad.append(f"{key}: no record in both raw directories")
                continue
            old, new = before[key], after[key]
            numbers = [float(value) for value in match.groups()[3:]]
            llmlip_before = old["timings"]["llmlip"]["median_ns"] / 1e6
            llmlip_after = new["timings"]["llmlip"]["median_ns"] / 1e6
            strided_before = total_ns(old) / 1e6
            strided_after = total_ns(new) / 1e6
            ratio_before = total_ns(old) / old["timings"]["llmlip"]["median_ns"]
            ratio_after = total_ns(new) / new["timings"]["llmlip"]["median_ns"]
            speedup = total_ns(old) / total_ns(new)
            speedups.append(speedup)
            label = f"{cell} {dtype} {threads}T {atoms}"
            close(f"{label} llmlip before", numbers[0], llmlip_before, MS_TOLERANCE, bad)
            close(f"{label} strided before", numbers[1], strided_before, MS_TOLERANCE, bad)
            close(f"{label} ratio before", numbers[2], ratio_before, RATIO_TOLERANCE, bad)
            close(f"{label} llmlip after", numbers[3], llmlip_after, MS_TOLERANCE, bad)
            close(f"{label} strided after", numbers[4], strided_after, MS_TOLERANCE, bad)
            if cell == "cg-vjp":
                split = new["timings"]["strided_split_total_ns"] / 1e6
                close(f"{label} split after", numbers[5], split, MS_TOLERANCE, bad)
                close(f"{label} ratio after", numbers[6], ratio_after, RATIO_TOLERANCE, bad)
                close(f"{label} speedup", numbers[7], speedup, MS_TOLERANCE, bad)
            else:
                close(f"{label} ratio after", numbers[5], ratio_after, RATIO_TOLERANCE, bad)
                close(f"{label} speedup", numbers[6], speedup, MS_TOLERANCE, bad)
            checked += 1
    missing = set(after) - rows
    extra = rows - set(after)
    if missing or extra:
        bad.append(f"table rows: missing {sorted(missing)}, unknown {sorted(extra)}")
    low, high = speedup_range
    if not speedups:
        bad.append("no table rows found")
    elif abs(min(speedups) - low) > 0.005 or abs(max(speedups) - high) > 0.005:
        bad.append(
            f"speedups run {min(speedups):.2f}-{max(speedups):.2f}x, "
            f"the prose says {low:g}-{high:g}x"
        )
    return checked


def check_targets(text: str, after: dict, bad: list[str]) -> int:
    """§1: the f32 4T VJP ratio at 128 and 1024 atoms."""
    checked = 0
    for atoms in (128, 1024):
        key = ("cg-vjp", "f32", 4, atoms)
        record = after.get(key)
        if record is None:
            bad.append(f"{key}: no after record")
            continue
        ratio = total_ns(record) / record["timings"]["llmlip"]["median_ns"]
        target = f"atoms | <= 1.00 | **{ratio:.3f}**"
        if target not in text:
            bad.append(f"§1 does not claim {ratio:.3f} for f32 4T VJP at {atoms} atoms")
        checked += 1
    return checked


def check_anchors(text: str, bad: list[str]) -> int:
    """§3: each campaign's anchor cell at both ends, and its drift."""
    checked = 0
    rows = {match.group(1): match for match in ANCHOR_ROW.finditer(text)}
    if set(rows) != {"before", "after"}:
        bad.append("§3 does not hold one anchor row per campaign")
    for name, raw_dir in (("before", BEFORE), ("after", AFTER)):
        match = rows.get(name)
        if match is None:
            continue
        with open(os.path.join(raw_dir, "campaign.txt")) as handle:
            payloads = [json.loads(line) for line in handle if line.startswith("{")]
        # 16 cells plus the two anchors: the first and the last payload are the
        # anchors, the middle ones are the cells.
        if len(payloads) != 18:
            bad.append(f"{name} campaign.txt holds {len(payloads)} records, expected 18")
            continue
        for side, index in (("strided", 0), ("llmlip", 1)):
            read = (
                (lambda r: r["timings"]["strided"]["median_ns"])
                if side == "strided"
                else (lambda r: r["timings"]["llmlip"]["median_ns"])
            )
            start = read(payloads[0]) / 1e6
            end = read(payloads[-1]) / 1e6
            drift = 100 * abs(end - start) / max(start, end)
            base = 2 if side == "strided" else 5
            close(
                f"{name} {side} anchor start",
                float(match.group(base)),
                start,
                ANCHOR_TOLERANCE,
                bad,
            )
            close(
                f"{name} {side} anchor end",
                float(match.group(base + 1)),
                end,
                ANCHOR_TOLERANCE,
                bad,
            )
            close(
                f"{name} {side} drift",
                float(match.group(base + 2)),
                drift,
                ANCHOR_TOLERANCE,
                bad,
            )
            checked += 1
    return checked


def check_allocation(text: str, before: dict, after: dict, bad: list[str]) -> int:
    """§6: no fused call allocates, and the buffer sizes it quotes."""
    checked = 0
    for key, record in after.items():
        if key[0] != "cg-vjp":
            continue
        timings = record["timings"]["strided_vjp_fused"]
        if timings["alloc_calls_total"] != 0 or timings["alloc_bytes_total"] != 0:
            bad.append(f"{key}: the fused call allocated {timings['alloc_bytes_total']} B")
        checked += 1
    # §6's buffer table, and the "unchanged within 1 MiB" claim over every cell.
    rows = 0
    for match in BUFFER_ROW.finditer(text):
        dtype, atoms = match.group(1), int(match.group(2))
        claimed = (float(match.group(3)), float(match.group(4)))
        record = after[("cg-vjp", dtype, 1, atoms)]
        got = (record["grad_x_mib"], record["bar_y_intermediate_mib"])
        for name, want, have in zip(("grad_x", "bar_y intermediate"), claimed, got):
            if abs(want - have) > 0.05:
                bad.append(f"{dtype} {atoms} atoms {name}: report {want}, records {have}")
            checked += 1
        rows += 1
    if rows != 4:
        bad.append(f"§6 has {rows} buffer rows, expected 4")
    if "unchanged within 1 MiB" not in text:
        bad.append("§6 no longer claims the peak RSS stays within 1 MiB")
    largest = max(
        (abs(after[key]["rss_bytes"] - before[key]["rss_bytes"]) for key in after),
        default=0,
    )
    if largest > 2**20:
        bad.append(f"peak RSS moved {largest / 2**20:.2f} MiB, the report says within 1 MiB")
    checked += 1
    return checked


def check_agreement(text: str, after: dict, bad: list[str]) -> int:
    """§7: the worst scale-relative disagreement per dtype and gradient."""
    checked = 0
    for dtype in ("f64", "f32"):
        cells = [record for key, record in after.items() if key[0] == "cg-vjp" and key[1] == dtype]
        if len(cells) != 4:
            bad.append(f"{dtype}: {len(cells)} VJP cells, expected 4")
            continue
        worst = [
            max(cell["check"][f"bar_{name}_rel_to_scale"] for cell in cells)
            for name in ("x", "w", "y")
        ]
        for name, value in zip("xwy", worst):
            if f"{value:.2e}" not in text:
                bad.append(f"§7 does not quote bar_{name} {value:.2e} for {dtype}")
            checked += 1
    return checked


def main() -> int:
    worklog = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKLOG
    with open(worklog) as handle:
        text = handle.read()
    before = load(BEFORE)
    after = load(AFTER)
    if len(before) != 16 or len(after) != 16:
        print(f"raw directories hold {len(before)} and {len(after)} records, expected 16 each")
        return 1
    bad: list[str] = []
    cells = check_cells(text, before, after, bad, (1.27, 2.23))
    targets = check_targets(text, after, bad)
    anchors = check_anchors(text, bad)
    allocations = check_allocation(text, before, after, bad)
    agreement = check_agreement(text, after, bad)
    if bad:
        for line in bad:
            print(f"MISMATCH {line}")
        return 1
    print(
        f"ok: {cells} table rows, {targets} targets, {anchors} anchors, "
        f"{allocations} allocation/buffer cells, {agreement} agreement figures"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
