#!/usr/bin/env python3
"""Count the strided-rs audit baseline rows (docs/worklogs/2026-11-08-issue-216-*-baseline.md).

The worklog states one rule per row; this file is the only implementation of
those rules. Run it as `python3 scripts/count-audit-metrics.py .` from the
repository root. `scripts/test-count-audit-metrics.py` pins the cases a naive
line scan gets wrong: block comments, line-spanning raw strings, `#[cfg(test)]
mod name;` declarations, and `pub extern "C" fn`.

Rows (production scope = `strided-*/src/**/*.rs`, `*/tests/` excluded, inline
`#[cfg(test)] mod name { .. }` bodies skipped):

  invariant               `//` line comments (doc comments excluded) containing the marker
  safety                  the same for `// SAFETY:`; an independent total, not paired with `unsafe`
  unsafe                  occurrences of `unsafe` + `{` in code
  pub_result / with_errors a `pub` `fn` whose signature (to the first `{` or `;`) returns `Result<`,
                          and how many of those carry a `/// # Errors` line
  inline_mods             `#[cfg(test)] mod name { .. }` blocks (declarations with `;` do not count)
  files_over_1000         `wc -l` over `strided-*/**/*.rs`
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PRODUCTION_GLOB = "strided-*/src/**/*.rs"
ALL_SOURCE_GLOB = "strided-*/**/*.rs"

FUNCTION_RE = re.compile(r'\s*pub (?:[\w"\']+ )*fn ')
INLINE_TEST_MOD_RE = re.compile(r"(?:pub\s+)?mod\s+\w+\s*\{")
INLINE_TEST_DECL_RE = re.compile(r"(?:pub\s+)?mod\s+\w+\s*;")
RAW_STRING_PREFIX_RE = re.compile(r"\b(?:br|r)(#*)\"")
UNSAFE_RE = re.compile(r"\bunsafe\s*\{")


def scan_line(raw: str, state: dict[str, object]) -> tuple[str, str, str]:
    """Return (code, code_without_strings, line_comment) for one line.

    State carries open block comments (Rust nests them) and an open raw string
    across lines. Both returned strings keep the line's length and column
    positions, so `^\\s*pub`, brace counting and offsets stay valid.
    """
    no_comments: list[str] = []
    no_strings: list[str] = []
    comment = ""
    i = 0
    depth = int(state["block_comment"])  # type: ignore[arg-type]
    raw_hashes = state["raw_string"]  # type: ignore[assignment]
    while i < len(raw):
        if depth:
            if raw.startswith("/*", i):
                depth += 1
                i += 2
                continue
            if raw.startswith("*/", i):
                depth -= 1
                i += 2
                continue
            no_comments.append(" ")
            no_strings.append(" ")
            i += 1
            continue
        if raw_hashes is not None:
            closing = '"' + "#" * int(raw_hashes)  # type: ignore[arg-type]
            end = raw.find(closing, i)
            if end < 0:
                no_comments.append(" " * (len(raw) - i))
                no_strings.append(" " * (len(raw) - i))
                i = len(raw)
                continue
            span = raw[i:end + len(closing)]
            no_comments.append(span)
            no_strings.append(" " * len(span))
            i = end + len(closing)
            raw_hashes = None
            continue
        if raw.startswith("//", i):
            comment = raw[i:]
            no_comments.append(" " * len(comment))
            no_strings.append(" " * len(comment))
            break
        if raw.startswith("/*", i):
            depth += 1
            no_comments.append("  ")
            no_strings.append("  ")
            i += 2
            continue
        prefix = RAW_STRING_PREFIX_RE.match(raw, i)
        if prefix:
            hashes = len(prefix.group(1))
            opening = prefix.end() - i
            closing = '"' + "#" * hashes
            end = raw.find(closing, i + opening)
            if end < 0:
                span = raw[i:]
                no_comments.append(span)
                no_strings.append(" " * len(span))
                i = len(raw)
                raw_hashes = hashes
                continue
            span = raw[i:end + len(closing)]
            no_comments.append(span)
            no_strings.append(" " * len(span))
            i = end + len(closing)
            continue
        if raw[i] == '"':
            j = i + 1
            while j < len(raw):
                if raw[j] == "\\":
                    j += 2
                    continue
                if raw[j] == '"':
                    j += 1
                    break
                j += 1
            span = raw[i:j]
            no_comments.append(span)
            no_strings.append(" " * len(span))
            i = j
            continue
        if raw[i] == "'" and i + 2 < len(raw) and raw[i + 1] != "\\" and raw[i + 2] == "'":
            no_comments.append("'''")
            no_strings.append("   ")
            i += 3
            continue
        no_comments.append(raw[i])
        no_strings.append(raw[i])
        i += 1
    state["block_comment"] = depth
    state["raw_string"] = raw_hashes
    return "".join(no_comments), "".join(no_strings), comment


def scan_file(text: str) -> list[dict[str, object]]:
    """Split a file into rows of (code, code_without_strings, line_comment, doc_comment)."""
    state: dict[str, object] = {"block_comment": 0, "raw_string": None}
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        head = raw.lstrip()
        if head.startswith("///") or head.startswith("//!"):
            rows.append(
                {
                    "code": " " * len(raw),
                    "code_no_strings": " " * len(raw),
                    "comment": "",
                    "doc": raw,
                }
            )
            continue
        if head.startswith("//") and not state["block_comment"] and state["raw_string"] is None:
            rows.append(
                {
                    "code": " " * len(raw),
                    "code_no_strings": " " * len(raw),
                    "comment": raw,
                    "doc": "",
                }
            )
            continue
        no_comments, no_strings, comment = scan_line(raw, state)
        rows.append(
            {
                "code": no_comments,
                "code_no_strings": no_strings,
                "comment": comment,
                "doc": "",
            }
        )
    return rows


def drop_test_bodies(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove `#[cfg(test)] mod name { .. }` bodies; a `mod name;` declaration removes nothing."""
    kept: list[dict[str, object]] = []
    i = 0
    while i < len(rows):
        code = str(rows[i]["code"]).strip()
        if code != "#[cfg(test)]":
            kept.append(rows[i])
            i += 1
            continue
        j = i + 1
        while j < len(rows) and (
            not str(rows[j]["code"]).strip() or str(rows[j]["code"]).strip().startswith("#[")
        ):
            j += 1
        if j == len(rows):
            kept.append(rows[i])
            i += 1
            continue
        item = str(rows[j]["code"])
        if INLINE_TEST_DECL_RE.match(item.strip()):
            # `#[cfg(test)] mod tests;` — the body lives in another file, nothing to skip.
            kept.append(rows[i])
            i += 1
            continue
        if not INLINE_TEST_MOD_RE.match(item.strip()):
            kept.append(rows[i])
            i += 1
            continue
        depth = 0
        k = j
        while k < len(rows):
            depth += str(rows[k]["code"]).count("{") - str(rows[k]["code"]).count("}")
            k += 1
            if depth <= 0 and "}" in str(rows[k - 1]["code"]):
                break
        i = k
    return kept


def count(root: Path) -> dict[str, int]:
    production = [
        p for p in sorted(root.glob(PRODUCTION_GLOB)) if "/tests/" not in str(p)
    ]
    all_source = [p for p in sorted(root.glob(ALL_SOURCE_GLOB)) if "/target/" not in str(p)]
    rows = {p: drop_test_bodies(scan_file(p.read_text(encoding="utf-8"))) for p in production}

    invariant = safety = unsafe = 0
    for p in production:
        for row in rows[p]:
            comment = str(row["comment"])
            invariant += comment.count("// INVARIANT:")
            safety += comment.count("// SAFETY:")
            unsafe += len(UNSAFE_RE.findall(str(row["code_no_strings"])))

    pub_result = with_errors = 0
    for p in production:
        has_errors, attr_depth = False, 0
        for i, row in enumerate(rows[p]):
            if row["doc"]:
                has_errors = has_errors or str(row["doc"]).lstrip().startswith("/// # Errors")
                continue
            code = str(row["code"])
            stripped = code.strip()
            if attr_depth > 0:
                attr_depth += stripped.count("(") + stripped.count("[")
                attr_depth -= stripped.count(")") + stripped.count("]")
                continue
            if stripped.startswith("#["):
                balance = stripped.count("(") + stripped.count("[")
                balance -= stripped.count(")") + stripped.count("]")
                attr_depth = max(balance, 0)
                continue
            if not stripped:
                continue
            if FUNCTION_RE.match(code):
                signature = []
                for part in rows[p][i:]:
                    signature.append(re.split(r"[{;]", str(part["code"]), 1)[0])
                    if re.search(r"[{;]", str(part["code"])):
                        break
                flat = " ".join(signature)
                if "->" in flat and "Result<" in flat.split("->", 1)[1]:
                    pub_result += 1
                    with_errors += has_errors
            has_errors = False
    inline_mods = 0
    for p in all_source:
        rows = scan_file(p.read_text(encoding="utf-8"))
        for i, row in enumerate(rows):
            if str(row["code"]).strip() != "#[cfg(test)]":
                continue
            j = i + 1
            while j < len(rows) and str(rows[j]["code"]).strip().startswith("#["):
                j += 1
            if j < len(rows) and INLINE_TEST_MOD_RE.match(str(rows[j]["code"]).strip()):
                inline_mods += 1
    files_over_1000 = sum(
        1 for p in all_source if len(p.read_text(encoding="utf-8").splitlines()) > 1000
    )
    return {
        "invariant": invariant,
        "safety": safety,
        "unsafe": unsafe,
        "pub_result": pub_result,
        "with_errors": with_errors,
        "inline_mods": inline_mods,
        "files_over_1000": files_over_1000,
    }


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    counts = count(root)
    print("invariant", counts["invariant"])
    print("safety", counts["safety"])
    print("unsafe", counts["unsafe"])
    print("pub_result", counts["pub_result"], "with_errors", counts["with_errors"])
    print("inline_mods", counts["inline_mods"])
    print("files_over_1000", counts["files_over_1000"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
