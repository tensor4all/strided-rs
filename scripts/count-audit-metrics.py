#!/usr/bin/env python3
"""Approximate the strided-rs audit baseline rows with a line-oriented scan.

Rows, as printed (production scope = `strided-*/src/**/*.rs`, minus a `tests/`
component, a `tests.rs` file and every file only test-only `mod name;`
declarations load -- a file a production declaration loads too stays in -- with
inline `#[cfg(test)] mod name { .. }` bodies skipped):

  invariant               `//` line comments containing the marker
  safety                  the same for `// SAFETY:`; an independent total, not paired with `unsafe`
  unsafe                  occurrences of `unsafe` + `{` in the scanned text
  pub_result / with_errors a `pub` `fn` whose signature (to the first `{` or `;` outside
                          brackets) returns `Result<`, and how many of those carry a
                          `# Errors` heading (a heading that merely starts with
                          `Errors`, such as `# ErrorsAreNotDocumented`, is another
                          section)
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
- Module resolution follows the attribute and inline-module rules the fixtures in
  `scripts/test-count-audit-metrics.py` pin; an exotic `#[path]` that climbs out
  of the source tree with `..` is normalized, not rejected.
- A nested block-comment opener (`/*a/*b*/c*/`) is consumed without keeping its
  columns, so each nesting level shortens the returned line by two characters and
  the columns after it shift. No production source in this repository hits it.

Where a row matters, measure the thing itself instead of this script: run
`cargo test --doc` for the doctest row, and read a file for a per-block
`// SAFETY:` audit.
"""

from __future__ import annotations

import os
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
# A `# Errors` heading; `# ErrorsAreNotDocumented` is a different heading, and a
# `## Errors` subheading is not the section. Trailing prose is allowed because
# Markdown treats `# Errors are returned` as a heading that also says "Errors".
ERRORS_HEADING_RE = re.compile(r"///\s*# Errors(?:\s|$)")


def scan_line(raw: str, state: dict[str, object]) -> tuple[str, str, str]:
    """Return (code, code_without_strings, line_comment) for one line.

    State carries open block comments (Rust nests them) and an open raw string
    across lines. Both returned strings keep the line's length and column
    positions, so `^\\s*pub`, brace counting and offsets stay valid — except across
    a nested block-comment opener, where two characters are dropped per nesting
    level (see Known limits).
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
        # A `///` or `//!` encountered mid-line (a trailing doc comment) is still
        # documentation, not an ordinary comment.
        is_doc = comment.startswith("///") or comment.startswith("//!")
        rows.append(
            {
                "code": no_comments,
                "code_no_strings": no_strings,
                "comment": "" if is_doc else comment,
                "doc": comment if is_doc else "",
            }
        )
    return rows


def attribute_row_set(rows: list[dict[str, object]]) -> set[int]:
    """Rows that belong to an attribute, including its continuation lines.

    `#[allow(` / `dead_code` / `)]` is one attribute on three rows; the closing row
    does not start with `#[`, so a backward walk needs this map to recognize it.
    """
    covered: set[int] = set()
    i = 0
    while i < len(rows):
        stripped = str(rows[i]["code"]).strip()
        if not stripped.startswith("#["):
            i += 1
            continue
        balance = stripped.count("[") + stripped.count("(")
        balance -= stripped.count("]") + stripped.count(")")
        covered.add(i)
        i += 1
        while balance > 0 and i < len(rows):
            continuation = str(rows[i]["code"]).strip()
            balance += continuation.count("[") + continuation.count("(")
            balance -= continuation.count("]") + continuation.count(")")
            covered.add(i)
            i += 1
    return covered


def drop_test_bodies(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Remove `#[cfg(test)] mod name { .. }` bodies and the documentation they own.

    A `mod name;` declaration removes nothing here: its body lives in another file,
    and the module graph handles that file. Documentation and further attributes
    that precede the `#[cfg(test)]` line belong to the same item, so they are
    dropped with it -- otherwise the module's `# Errors` would be credited to the
    next production `fn`.
    """
    attribute_rows = attribute_row_set(rows)
    dropped: set[int] = set()
    i = 0
    while i < len(rows):
        if str(rows[i]["code"]).strip() != "#[cfg(test)]":
            i += 1
            continue
        j = skip_attributes(rows, i + 1)
        if j == len(rows):
            i += 1
            continue
        item = str(rows[j]["code"]).strip()
        if INLINE_TEST_DECL_RE.match(item) or not INLINE_TEST_MOD_RE.match(item):
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
        start = i
        while start > 0:
            previous = rows[start - 1]
            if (
                previous["doc"]
                or not str(previous["code"]).strip()
                or (start - 1) in attribute_rows
            ):
                # Blank lines do not end an attribute list in Rust, and a
                # multi-line attribute belongs to the item it precedes.
                start -= 1
                continue
            break
        dropped.update(range(start, k))
        i = k
    return [row for index, row in enumerate(rows) if index not in dropped]


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


def attribute_block(
    rows: list[dict[str, object]], i: int
) -> tuple[int, bool, str | None]:
    """Read the attribute list at `i`: (item row, `#[cfg(test)]` present, `#[path]` value)."""
    j = skip_attributes(rows, i)
    cfg_test = False
    path_value = None
    for k in range(i, j):
        text = str(rows[k]["code"]).strip()
        if re.fullmatch(r"#\[cfg\(test\)\]", text):
            cfg_test = True
        match = PATH_ATTR_RE.search(text)
        if match:
            path_value = match.group(1)
    return j, cfg_test, path_value


def module_declarations(
    rows: list[dict[str, object]],
) -> list[tuple[str, tuple[str, ...], str, bool, bool]]:
    """Every `mod NAME;` of a file as (base, directory, leaf, explicit, test-only).

    `directory` is the chain of inline `mod X { .. }` directories the declaration
    sits in, and `base` says where that chain starts: `"file"` for the declaring
    file's own directory, `"module"` for the directory Rust derives from the file
    name (`src/foo.rs` owns `src/foo/`). Both follow Rust's own resolution:

    * a plain `mod NAME;` resolves under the file's module directory plus the
      inline names;
    * an explicit `#[path]` at the top level resolves against the file's
      directory;
    * an explicit `#[path]` inside an inline module resolves against the
      enclosing module's directory, which the inline chain has already moved.

    A `#[path]` on an inline module moves the directory for its children to that
    path, so the chain carries it instead of the module's name. Test-only means
    the declaration carries `#[cfg(test)]` or sits inside an inline module that
    does.
    """
    declarations: list[tuple[str, tuple[str, ...], str, bool, bool]] = []
    # (base kind, directory components, brace depth, test-only) per inline module.
    stack: list[tuple[str, tuple[str, ...], int, bool]] = []
    depth = 0
    pending: tuple[bool, str | None] | None = None
    i = 0
    while i < len(rows):
        code = str(rows[i]["code_no_strings"])
        stripped = code.strip()
        if stripped.startswith("#["):
            j, cfg_test, path_value = attribute_block(rows, i)
            pending = (cfg_test, path_value)
            # The item line is processed next so its braces move `depth`.
            i = j if j < len(rows) else len(rows)
            continue
        cfg_test, path_value = pending if pending is not None else (False, None)
        pending = None
        enclosing_test = any(test for _, _, _, test in stack)
        base, directory = (stack[-1][0], stack[-1][1]) if stack else ("module", ())
        inline = INLINE_TEST_MOD_RE.match(stripped)
        declaration = INLINE_TEST_DECL_RE.match(stripped)
        if inline:
            name = inline.group(1)
            if path_value is None:
                components = directory + (name,)
                kind = base if stack else "module"
            else:
                components = directory + tuple(
                    part for part in Path(path_value).parts if part not in {"", "."}
                )
                kind = base if stack else "file"
            stack.append((kind, components, depth, cfg_test or enclosing_test))
        elif declaration:
            name = declaration.group(1)
            explicit = path_value is not None
            kind = base if stack else ("file" if explicit else "module")
            leaf = path_value if explicit else name
            declarations.append((kind, directory, leaf, explicit, cfg_test or enclosing_test))
        depth += code.count("{") - code.count("}")
        while stack and depth <= stack[-1][2]:
            stack.pop()
        i += 1
    return declarations


def declared_module_dir(path: Path) -> Path:
    """The directory Rust looks in for a `mod NAME;` declared in `path`.

    `src/lib.rs`, `src/main.rs` and `src/foo/mod.rs` own their own directory;
    `src/foo.rs` owns `src/foo/`, not `src/`.
    """
    if path.stem in {"lib", "main", "mod"}:
        return path.parent
    return path.parent / path.stem


def production_files(root: Path) -> list[Path]:
    """Production sources, minus every file only the test-only module graph owns.

    Test-only files seed the graph (`tests/` path components, `tests.rs`, and any
    target of a `#[cfg(test)] mod` declaration). A declaration inside a test-only
    file is test-only too, so the graph is closed to a fixed point over all
    sources: a test module's ordinary children (`support.rs` declaring
    `mod child;`) and its explicitly pathed children follow it out of production.
    Declarations resolve relative to the declaring module's directory plus the
    enclosing inline modules, and a `#[path = "..."]` resolves the way Rust does
    (see [`module_declarations`]), so a `#[path]` inside an inline module keeps
    the containing file's module directory.

    Two rules keep the production side from shrinking by accident:

    * a file loaded by both a test-only and a production declaration stays in
      production, so a test-only alias cannot erase code the baseline counts;
    * a plain `mod NAME;` loads exactly `NAME.rs` or `NAME/mod.rs`, so its
      directory only *excludes* the files nothing declares (the historical
      module-subtree rule) and never claims a file an inline module's explicit
      `#[path]` owns.
    """
    sources = [
        p for p in sorted(root.glob(ALL_SOURCE_GLOB)) if "target" not in relative_parts(p, root)
    ]
    parsed = {p: module_declarations(scan_file(p.read_text(encoding="utf-8"))) for p in sources}
    seeds = {p for p in sources if "tests" in relative_parts(p, root) or p.name == "tests.rs"}
    # A file loaded through `#[path]` resolves its own children beside itself, not
    # in a directory named after it.
    module_dirs: dict[Path, Path] = {}
    test_only = set(seeds)
    changed = True
    while changed:
        changed = False
        # (declaring file, loaded file, test-only edge)
        edges: list[tuple[Path, Path, bool]] = []
        # (module directory, declaring file, test-only declaration)
        directories: list[tuple[Path, Path, bool]] = []
        for p in sources:
            module_base = module_dirs.get(p, declared_module_dir(p))
            for kind, directory, leaf, explicit, cfg_test in parsed[p]:
                base = p.parent if kind == "file" else module_base
                if explicit:
                    target = Path(os.path.normpath(base.joinpath(*directory, leaf)))
                    owned = {target}
                    if module_dirs.get(target) != target.parent:
                        module_dirs[target] = target.parent
                        changed = True
                else:
                    module = base.joinpath(*directory, leaf)
                    owned = {
                        module.with_name(f"{leaf}.rs"),
                        module / "mod.rs",
                    }
                    directories.append((module, p, cfg_test))
                edges.extend((p, q, cfg_test) for q in owned if q in sources)
        # Greatest fixed point: a file is test-only when every declaration that
        # loads it is, or, for a file no declaration loads, when every module
        # directory that contains it belongs to a test-only declaration.
        loaded = {target for _, target, _ in edges}
        closure = set(seeds)
        while True:
            added: set[Path] = set()
            for target in sources:
                if target in closure:
                    continue
                incoming = [
                    (declarer, cfg_test)
                    for declarer, loaded_file, cfg_test in edges
                    if loaded_file == target
                ]
                if incoming:
                    if all(cfg_test or declarer in closure for declarer, cfg_test in incoming):
                        added.add(target)
                    continue
                if target not in loaded:
                    containing = [
                        (declarer, cfg_test)
                        for module, declarer, cfg_test in directories
                        if module in target.parents
                    ]
                    if containing and all(
                        cfg_test or declarer in closure for declarer, cfg_test in containing
                    ):
                        added.add(target)
            if not added:
                break
            closure |= added
        if closure != test_only:
            test_only = closure
            changed = True
    return [
        p
        for p in sorted(root.glob(PRODUCTION_GLOB))
        if p in sources and p not in test_only
    ]


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
                has_errors = has_errors or bool(ERRORS_HEADING_RE.match(str(row["doc"]).strip()))
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
