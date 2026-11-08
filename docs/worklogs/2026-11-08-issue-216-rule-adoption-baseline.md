# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured on this branch's tree @ `41cb04d` (2026-11-08; no Rust source differs from
`origin/main` @ `06cd777`) over the maintained crates
(`strided-traits`, `strided-view`, `strided-basic`, `strided-kernel`,
`strided-fused`, `strided-nn`, `strided-perm`). These are source counts, so they
depend on the tree, not on the machine.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 52 | 0 |
| `// SAFETY:` comments | 300 | 29 |
| production `unsafe {` occurrences | 556 | 418 |
| public `Result`-returning `fn` | 186 | 172 |
| …of those with a `# Errors` section | 186 | 9 |
| runnable doctests | 123 | 5 |
| inline `#[cfg(test)] mod` modules | 22 | 53 |
| files over 1000 lines | 21 | 13 |

Three of the rows are already at or near their target because the work landed
between the issue's opening count and this baseline: the `# Errors` row is
closed (the sweep of #381 left no public `Result`-returning `fn` without the
section), the doctests are the same sweep, and the inline-test drop is the move
of the inline unit test modules into `*/tests/` directories (#362). The gap the
remediation issues should take next is `SAFETY` coverage, and 21 files are over
1000 lines. Read the old numbers as history and these as the baseline.

`SAFETY` is counted as two independent totals, not as a coverage ratio: 556
production `unsafe` occurrences and 300 `// SAFETY:` comments. A ratio would
claim that each of those comments documents one of those occurrences, and this
counter does not establish that pairing; the shared rule requires one comment
per block, so the per-block audit is the next measurement, not this one.

## Counting rules

A recount is only comparable if it uses the same rule, so the rules and the
counter below are the same statement in two forms: each table row is one
`print` line, and the counter is the only implementation. Production scope is
`strided-*/src/**/*.rs` with `*/tests/` excluded and inline `#[cfg(test)] mod`
bodies skipped for the first five rows; doc-comment and file rows scan all of
`strided-*/**/*.rs` (`target/` excluded). Comments, doc comments and string
literals are removed from the source text before anything is counted, because
`rules/rust/index.md:23` counts production `unsafe` "precisely (skip
`// SAFETY:` comments, doc comments, string literals, tests, generated code)".
The doctest row is not in the counter: it is `cargo test --doc`, which is the
definition of "runnable"; the recorded value is the command's own output at this
tree (123 passed, 0 ignored).

| Row | Rule |
|-----|------|
| `// INVARIANT:` | `//` line comments (doc comments excluded) containing the marker |
| `// SAFETY:` | `//` line comments (doc comments excluded) containing the marker; an independent total, not paired with the row below |
| `unsafe {` | occurrences of `unsafe` + `{` in code, not blocks, after comments and string literals are removed; an independent total, not paired with the row above |
| public `Result` `fn` | a line starting a `pub` (or `pub unsafe` / `pub const` / `pub extern "C"`) `fn` whose signature, read to the first `{` or `;`, has a `Result<` after its `->` |
| `# Errors` | the doc block nearest above such a `fn` contains a `/// # Errors` line; attribute lines (`#[inline]`, `#[must_use]`, `#[allow(..)]`, including multi-line ones) and blank lines between the doc block and the `fn` do not break the association |
| runnable doctests | `cargo test --doc` pass count at the repository root, not a fence count |
| inline test modules | `#[cfg(test)]` immediately followed by `mod` |
| files over 1000 lines | `wc -l` over `strided-*/**/*.rs` |

```python
# python3 counter.py <repo-root>
import re, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
prod = [p for p in sorted(root.glob("strided-*/src/**/*.rs")) if "/tests/" not in str(p)]
allsrc = [p for p in sorted(root.glob("strided-*/**/*.rs")) if "/target/" not in str(p)]


def classify(text):
    """Per line: (code, line comment, doc comment). Code loses comments and literals."""
    rows = []
    for raw in text.splitlines():
        head = raw.lstrip()
        if head.startswith("///") or head.startswith("//!"):
            rows.append(("", "", raw))
            continue
        if head.startswith("//"):
            rows.append(("", raw, ""))
            continue
        code, buf, i = [], [], 0
        while i < len(raw):
            c = raw[i]
            if c == '"':
                j = i + 1
                while j < len(raw) and raw[j] != '"':
                    j += 2 if raw[j] == "\\" else 1
                buf.append('""')
                i = j + 1
                continue
            if c == "'" and i + 2 < len(raw) and raw[i + 1] != "\\" and raw[i + 2] == "'":
                buf.append("''")
                i += 3
                continue
            if raw.startswith("//", i):
                rows.append(("".join(buf + code), raw[i:], ""))
                break
            code.append(c)
            i += 1
        else:
            rows.append(("".join(buf + code), "", ""))
    return rows


def drop_inline_tests(rows):
    """Remove `#[cfg(test)] mod ... { .. }` bodies; the shared rule skips tests."""
    kept, i = [], 0
    while i < len(rows):
        if rows[i][0].strip() != "#[cfg(test)]":
            kept.append(rows[i])
            i += 1
            continue
        j = i + 1
        while j < len(rows) and (rows[j][0].strip() == "" or rows[j][0].strip().startswith("#[")):
            j += 1
        if j == len(rows) or not re.match(r"(?:pub\s+)?mod\s+\w+", rows[j][0].strip()):
            kept.append(rows[i])
            i += 1
            continue
        depth, k = 0, j
        while k < len(rows):
            depth += rows[k][0].count("{") - rows[k][0].count("}")
            k += 1
            if depth <= 0 and "}" in rows[k - 1][0]:
                break
        i = k
    return kept


rows = {p: drop_inline_tests(classify(p.read_text(encoding="utf-8"))) for p in prod}
print("invariant", sum(r[1].count("// INVARIANT:") for p in prod for r in rows[p]))
print("safety", sum(r[1].count("// SAFETY:") for p in prod for r in rows[p]))
print("unsafe", sum(len(re.findall(r"\bunsafe\s*\{", r[0])) for p in prod for r in rows[p]))

FN = re.compile(r'\s*pub (?:[\w"\']+ )*fn ')
pub = err = 0
for p in prod:
    has_errors, attr_depth = False, 0
    for i, row in enumerate(rows[p]):
        if row[2]:                               # doc comment line
            has_errors = has_errors or row[2].lstrip().startswith("/// # Errors")
            continue
        s = row[0].strip()
        if attr_depth > 0:                       # inside a multi-line attribute
            attr_depth += s.count("(") + s.count("[") - s.count(")") - s.count("]")
            continue
        if s.startswith("#["):
            attr_depth = max(s.count("(") + s.count("[") - s.count(")") - s.count("]"), 0)
            continue
        if s == "" or s.startswith("//"):
            continue                             # keeps the doc block associated
        if FN.match(row[0]):
            sig = []                             # signature runs to the first `{` or `;`
            for part in rows[p][i:]:
                sig.append(re.split(r"[{;]", part[0], 1)[0])
                if re.search(r"[{;]", part[0]):
                    break
            flat = " ".join(sig)
            if "->" in flat and "Result<" in flat.split("->", 1)[1]:
                pub += 1
                err += has_errors
        has_errors = False                       # any other line starts a new item
print("pub_result", pub, "with_errors", err)
inline = sum(
    len(re.findall(r"#\[cfg\(test\)\]\s*\n\s*mod ", p.read_text(encoding="utf-8")))
    for p in allsrc
)
print("inline_mods", inline)
print("files_over_1000", sum(1 for p in allsrc if len(p.read_text(encoding="utf-8").splitlines()) > 1000))
```

Three rules were tightened after review, and they move the numbers:

- The signature is read to the first `{` or `;`, not to a fixed number of
  lines. A signature whose return type sits past any fixed window drops out of
  both the denominator and the numerator; `strided-basic/src/copy_plan.rs:503`
  (`execute`) is thirteen lines long and was one of the functions the old
  eight-line window missed, so the denominator was 148 instead of 186.
- The doc block is found across attributes. Scanning only the lines directly
  above the `fn` missed a documented entry with an `#[inline]` between the doc
  comment and the item, such as
  `strided-basic/src/execution.rs:132` (`for_each_inner_block_preordered`).
  Before the fix the numerator was 138; both fixes together give 186 of 186.
- The `// SAFETY:` and `unsafe {` rows are counted on code with comments, doc
  comments and string literals removed, and with inline `#[cfg(test)] mod`
  bodies skipped. Counting raw text mixed 29 doc-comment `// SAFETY:` markers
  into the comment total and 39 doc-comment `unsafe {` lines into the
  occurrence total, which is why the previous numbers were 332 and 598.
  They are now 300 and 556, reported as two totals rather than a coverage
  ratio: no pairing between a comment and a block is established here.
