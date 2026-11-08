#!/usr/bin/env python3
"""Approximate the strided-rs audit baseline rows with a line-oriented scan.

Rows, as printed (production scope = `strided-*/src/**/*.rs`, minus a `tests/`
component, a `tests.rs` file and every file a `#[cfg(test)] mod name;`
declaration makes test-only, with inline `#[cfg(test)] mod name { .. }` bodies
skipped):

  invariant               `//` line comments containing the marker
  safety                  the same for `// SAFETY:`; an independent total, not paired with `unsafe`
  unsafe                  occurrences of `unsafe` + `{` in the scanned text
  pub_result / with_errors a `pub` `fn` whose signature (to the first `{` or `;` outside
                          brackets) returns `Result<`, and how many of those carry a
                          `/// # Errors` line
  inline_mods             `#[cfg(test)] mod name { .. }` blocks (a `mod name;` declaration is 0)
  files_over_1000         logical lines (`str.splitlines()`) over `strided-*/**/*.rs`

Run it as `python3 scripts/count-audit-metrics.py .` from the repository root;
`scripts/test-count-audit-metrics.py` pins the cases a plain line scan gets wrong.

# Known limits: an approximation, not a Rust parser

The scanner tracks `/* */` (nested), line comments, char literals and string
literals including `r#"…"#` and ordinary `"…"` across lines. Everything else is
line-oriented, and the following are known to be miscounted. Do not read a row as
exact, and do not quote one as proof that a file or a tree is clean.

The `// INVARIANT:`, `// SAFETY:` and `unsafe {` rows count text the scan believes
is code, and text it misclassifies is dropped rather than invented, so they are a
**lower bound** for the source text in scope: the true count is at least what is
printed. The `Result` rows are a lower bound in the same way, except that a `<` or
`>` used as a comparison operator inside a signature can move an item either way
(see the last limit below); an over-count anywhere else is a bug here, not a
licence to read a row as exact. A count of zero therefore means "the scan found
none", not "none exists".

- A `macro_rules!` template is scanned where it is written, so a template that
  spells `pub fn $name(..) -> Result<..>` counts once; its invocations are not
  expanded, so an item that only the expansion creates is never seen.
- A string literal inside an attribute (`#[doc = "("]`) keeps its punctuation in
  the attribute's bracket balance, so the attribute can read as unfinished and the
  item below it is not seen.
- Braces inside a macro body that the scanner reads as code (not as a string)
  still move the `#[cfg(test)] mod` skip and the inline-module count.
- `unsafe` is counted as the text `unsafe` + `{`, so `unsafe extern` blocks,
  `unsafe fn` bodies whose brace sits on another item, and a macro that expands
  to `unsafe {` are not distinguished.
- `read_signature` treats `<` and `>` as brackets, so a signature with a
  comparison operator reads to the wrong place.
- A `mod NAME;` whose declaration is itself produced by a macro is not followed,
  and a `#[path = "..."]` given through a macro is not seen.

Where a row matters, measure the thing itself instead of this script: run
`cargo test --doc` for the doctest row, and read a file for a per-block
`// SAFETY:` audit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PRODUCTION_GLOB = "strided-*/src/**/*.rs"
ALL_SOURCE_GLOB = "strided-*/**/*.rs"

FUNCTION_RE = re.compile(r'\s*pub (?:[\w"\']+ )*fn ')
INLINE_TEST_MOD_RE = re.compile(r"(?:pub\s+)?mod\s+(\w+)\s*\{")
PATH_ATTR_RE = re.compile(r'#\[path\s*=\s*"([^"]+)"\s*\]')
INLINE_TEST_DECL_RE = re.compile(r"(?:pub\s+)?mod\s+(\w+)\s*;")
RAW_STRING_PREFIX_RE = re.compile(r"\b(?:br|cr|r)(#*)\"")
# A char literal, including the escaped forms `'\"'`, `'\\'` and `'\u{1F600}'`.
# A lifetime (`'a`) has no closing quote and does not match.
CHAR_LITERAL_RE = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F_]{1,6}\}|.)|[^\\'])'")
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
    plain_open = bool(state.get("plain_string"))
    while i < len(raw):
        if plain_open:
            # An ordinary `"` string left open on the previous line. Blank the
            # text up to its closing quote, honouring `\` escapes.
            end = i
            while end < len(raw):
                if raw[end] == "\\":
                    end += 2
                    continue
                if raw[end] == '"':
                    end += 1
                    plain_open = False
                    break
                end += 1
            end = min(end, len(raw))
            no_comments.append(" " * (end - i))
            no_strings.append(" " * (end - i))
            i = end
            continue
        if depth:
            if raw.startswith("/*", i):
                depth += 1
                i += 2
                continue
            if raw.startswith("*/", i):
                depth -= 1
                no_comments.append("  ")
                no_strings.append("  ")
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
            no_comments.append(" " * len(span))
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
            closed = False
            while j < len(raw):
                if raw[j] == "\\":
                    j += 2
                    continue
                if raw[j] == '"':
                    j += 1
                    closed = True
                    break
                j += 1
            if not closed:
                plain_open = True
            span = raw[i:j]
            no_comments.append(span)
            no_strings.append(" " * len(span))
            i = j
            continue
        literal = CHAR_LITERAL_RE.match(raw, i)
        if literal:
            span = literal.group(0)
            no_comments.append(span)
            no_strings.append(" " * len(span))
            i = literal.end()
            continue
        no_comments.append(raw[i])
        no_strings.append(raw[i])
        i += 1
    state["block_comment"] = depth
    state["raw_string"] = raw_hashes
    state["plain_string"] = plain_open
    return "".join(no_comments), "".join(no_strings), comment


def scan_file(text: str) -> list[dict[str, object]]:
    """Split a file into rows of (code, code_without_strings, line_comment, doc_comment)."""
    state: dict[str, object] = {"block_comment": 0, "raw_string": None, "plain_string": False}
    rows: list[dict[str, object]] = []
    for raw in text.splitlines():
        head = raw.lstrip()
        if (head.startswith("///") or head.startswith("//!")) and not (
            state["plain_string"] or state["block_comment"] or state["raw_string"] is not None
        ):
            rows.append(
                {
                    "code": " " * len(raw),
                    "code_no_strings": " " * len(raw),
                    "comment": "",
                    "doc": raw,
                }
            )
            continue
        if (
            head.startswith("//")
            and not state["block_comment"]
            and state["raw_string"] is None
            and not state["plain_string"]
        ):
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
        j = skip_attributes(rows, i + 1)
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
            text = str(rows[k]["code_no_strings"])
            depth += text.count("{") - text.count("}")
            k += 1
            if depth <= 0 and "}" in str(rows[k - 1]["code"]):
                break
        i = k
    return kept


def read_signature(rows: list[dict[str, object]], start: int) -> str:
    """Read a `fn` signature to its first `{` or `;` outside brackets.

    A `;` inside `[u8; 4]`, a generic parameter list or a `where`-clause
    lifetime is not the end of the item; only a `;` at bracket depth zero is.
    An arrow is skipped as one token, so the `>` of a callback's `->` cannot
    close a bracket the parameter list opened.
    """
    depth = 0
    parts: list[str] = []
    for row in rows[start:]:
        # Character literals such as `b'>'` are data, not brackets.
        code = str(row["code_no_strings"])
        index = 0
        while index < len(code):
            if code.startswith("->", index):
                index += 2
                continue
            char = code[index]
            if char in "([<{":
                if char == "{" and depth == 0:
                    parts.append(code[:index])
                    return " ".join(parts)
                depth += 1
            elif char in ")]>}":
                depth = max(depth - 1, 0)
            elif char == ";" and depth == 0:
                parts.append(code[:index])
                return " ".join(parts)
            index += 1
        parts.append(code)
    return " ".join(parts)


def returns_result(flat: str) -> bool:
    """True when the signature's own return type starts with `Result<`.

    Every arrow inside the parameter list (a callback type such as
    `fn() -> Result<..>`) sits at bracket depth >= 1, so only depth-zero arrows
    are the item's return type. An arrow is consumed as one token: counting its
    `>` as a closer would drop the depth of the enclosing parameter list and
    expose a callback's arrow as if it were the item's.
    """
    depth = 0
    index = 0
    while index < len(flat):
        if flat.startswith("->", index):
            if depth == 0:
                rest = flat[index + 2 :].lstrip()
                return rest.startswith("Result<")
            index += 2
            continue
        if (
            depth == 0
            and flat.startswith("where", index)
            and not (index > 0 and (flat[index - 1].isalnum() or flat[index - 1] == "_"))
            and not (index + 5 < len(flat) and (flat[index + 5].isalnum() or flat[index + 5] == "_"))
        ):
            # Past a top-level `where`, an arrow belongs to a bound
            # (`F: Fn() -> ..`), not to the item.
            return False
        char = flat[index]
        if char in "([<":
            depth += 1
        elif char in ")]>":
            depth = max(depth - 1, 0)
        index += 1
    return False


def relative_parts(path: Path, root: Path) -> tuple[str, ...]:
    """Repo-relative path components.

    The exclusions below must not look at the absolute path: a checkout under
    `~/target/...` or `~/tests/...` would otherwise drop every file.
    """
    try:
        return path.relative_to(root).parts
    except ValueError:
        return path.parts


def skip_attributes(rows: list[dict[str, object]], i: int) -> int:
    """Index of the first row after the attribute list starting at `i`.

    An attribute can span lines (`#[allow(` / `dead_code` / `)]`), so its brackets
    are balanced instead of only lines starting with `#[` being skipped.
    """
    while i < len(rows):
        stripped = str(rows[i]["code"]).strip()
        if not stripped:
            i += 1
            continue
        if not stripped.startswith("#["):
            return i
        balance = stripped.count("[") + stripped.count("(") - stripped.count("]") - stripped.count(")")
        i += 1
        while balance > 0 and i < len(rows):
            continuation = str(rows[i]["code"]).strip()
            balance += (
                continuation.count("[")
                + continuation.count("(")
                - continuation.count("]")
                - continuation.count(")")
            )
            i += 1
    return i


def inline_test_module_names(rows: list[dict[str, object]]) -> set[str]:
    """Module names a `#[cfg(test)] mod NAME { .. }` declares inline."""
    names: set[str] = set()
    i = 0
    while i < len(rows):
        if str(rows[i]["code"]).strip() != "#[cfg(test)]":
            i += 1
            continue
        j = skip_attributes(rows, i + 1)
        if j < len(rows):
            match = INLINE_TEST_MOD_RE.match(str(rows[j]["code"]).strip())
            if match:
                names.add(match.group(1))
        i += 1
    return names


def test_only_module_names(rows: list[dict[str, object]]) -> set[str]:
    """Module names a `#[cfg(test)] mod NAME;` declaration moves to another file.

    Further attributes may sit between the `#[cfg(test)]` line and the
    declaration, as in `strided-perm/src/lib.rs`'s `mod kernel;`.
    """
    names: set[str] = set()
    i = 0
    while i < len(rows):
        if str(rows[i]["code"]).strip() != "#[cfg(test)]":
            i += 1
            continue
        j = skip_attributes(rows, i + 1)
        if j < len(rows):
            match = INLINE_TEST_DECL_RE.match(str(rows[j]["code"]).strip())
            if match:
                explicit = None
                for k in range(i + 1, j):
                    path_match = PATH_ATTR_RE.search(str(rows[k]["code"]))
                    if path_match:
                        explicit = path_match.group(1)
                # `@relative/path` marks a `#[path = "..."]` target.
                names.add(f"@{explicit}" if explicit else match.group(1))
        i += 1
    return names


def declared_module_dir(path: Path) -> Path:
    """The directory Rust looks in for a `mod NAME;` declared in `path`.

    `src/lib.rs`, `src/main.rs` and `src/foo/mod.rs` own their own directory;
    `src/foo.rs` owns `src/foo/`, not `src/`.
    """
    if path.stem in {"lib", "main", "mod"}:
        return path.parent
    return path.parent / path.stem


def production_files(root: Path) -> list[Path]:
    """Production sources, minus the module subtree of every test-only `mod NAME;`.

    A path exclusion cannot see `#[cfg(test)] mod NAME;` in the declaring file, so
    the declaration is resolved the way Rust resolves it — relative to the
    declaring module's directory — and the whole subtree (`NAME.rs`, `NAME/mod.rs`
    and anything below `NAME/`) is excluded. A `#[path = "..."]` attribute instead
    names its file relative to the declaring file's own directory, so it can point
    outside that subtree. The exclusion set is collected from every candidate,
    including candidates that are themselves excluded, so one pass is enough.
    """
    candidates = [
        p
        for p in sorted(root.glob(PRODUCTION_GLOB))
        if "tests" not in relative_parts(p, root) and p.name != "tests.rs"
    ]
    excluded: set[Path] = set()
    for p in candidates:
        rows = scan_file(p.read_text(encoding="utf-8"))
        module_dir = declared_module_dir(p)
        for name in test_only_module_names(rows):
            if name.startswith("@"):
                # A `#[path = "..."]` target is relative to the file's own
                # directory, even when the declaration sits in `foo.rs`.
                excluded.add(p.parent / name[1:])
            else:
                excluded.add(module_dir / name)
        for name in inline_test_module_names(rows):
            excluded.add(module_dir / name)
    if not excluded:
        return candidates

    def is_excluded(path: Path) -> bool:
        for target in excluded:
            if path == target or path == target.with_name(f"{target.name}.rs") or target in path.parents:
                return True
        return False

    return [p for p in candidates if not is_excluded(p)]


def count(root: Path) -> dict[str, int]:
    production = production_files(root)
    all_source = [
        p for p in sorted(root.glob(ALL_SOURCE_GLOB)) if "target" not in relative_parts(p, root)
    ]
    rows = {p: drop_test_bodies(scan_file(p.read_text(encoding="utf-8"))) for p in production}

    invariant = safety = unsafe = 0
    for p in production:
        for row in rows[p]:
            comment = str(row["comment"])
            invariant += comment.count("// INVARIANT:")
            safety += comment.count("// SAFETY:")
        # Join the stripped lines before matching: `unsafe` and its `{` are often
        # split across lines, and a per-line match loses every such block.
        code = "\n".join(str(row["code_no_strings"]) for row in rows[p])
        unsafe += len(UNSAFE_RE.findall(code))

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
                flat = read_signature(rows[p], i)
                if returns_result(flat):
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
            j = skip_attributes(rows, i + 1)
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
