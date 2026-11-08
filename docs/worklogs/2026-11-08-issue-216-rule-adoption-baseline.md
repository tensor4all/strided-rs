# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured on `origin/main` @ `06cd777` (2026-11-08) over the maintained crates
(`strided-traits`, `strided-view`, `strided-basic`, `strided-kernel`,
`strided-fused`, `strided-nn`, `strided-perm`). These are source counts, so they
depend on the tree, not on the machine.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 57 | 0 |
| `// SAFETY:` comments | 332 | 29 |
| production `unsafe {` occurrences | 598 | 418 |
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
remediation issues should take next is `SAFETY` coverage: 332 of 598 production
`unsafe` occurrences carry a `// SAFETY:` comment, and 21 files are over 1000
lines. Read the old numbers as history and these as the baseline.

## Counting rules

A recount is only comparable if it uses the same rule, so the rules and the
counter below are the same statement in two forms: each table row is one
`print` line, and the counter is the only implementation. Production scope is
`strided-*/src/**/*.rs` with `*/tests/` excluded for the first five rows;
doc-comment and file rows scan all of `strided-*/**/*.rs` (`target/` excluded).
The doctest row is not in the counter: it is `cargo test --doc`, which is the
definition of "runnable".

| Row | Rule |
|-----|------|
| `// INVARIANT:` | lines containing the marker |
| `// SAFETY:` | lines containing the marker |
| `unsafe {` | occurrences of `unsafe` + `{`, not blocks (the shared rule wants a precise production count; this row is an upper bound that moves with the tree) |
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
lines = {p: p.read_text(encoding="utf-8").splitlines() for p in prod}
print("invariant", sum(t.count("// INVARIANT:") for ls in lines.values() for t in ls))
print("safety", sum(t.count("// SAFETY:") for ls in lines.values() for t in ls))
print("unsafe", sum(len(re.findall(r"\bunsafe\s*\{", "\n".join(ls))) for ls in lines.values()))
FN = re.compile(r'\s*pub (?:[\w"\']+ )*fn ')
pub = err = 0
for ls in lines.values():
    has_errors, attr_depth = False, 0
    for i, line in enumerate(ls):
        s = line.strip()
        if attr_depth > 0:                       # inside a multi-line attribute
            attr_depth += s.count("(") + s.count("[") - s.count(")") - s.count("]")
            continue
        if s.startswith("#["):
            balance = s.count("(") + s.count("[") - s.count(")") - s.count("]")
            attr_depth = max(balance, 0)         # 0 for a single-line attribute
            continue
        if s.startswith("///"):
            has_errors = has_errors or s.startswith("/// # Errors")
            continue
        if s == "" or s.startswith("//"):
            continue                             # keeps the doc block associated
        if FN.match(line):
            sig = []                             # signature runs to the first `{` or `;`
            for part in ls[i:]:
                sig.append(re.split(r"[{;]", part, 1)[0])
                if re.search(r"[{;]", part):
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

Two rules were tightened after review, and they move the numbers:

- The signature is read to the first `{` or `;`, not to a fixed number of
  lines. A signature whose return type sits past any fixed window drops out of
  both the denominator and the numerator; `strided-basic/src/copy_plan.rs:503`
  (`execute`) is nine lines long and was one of the 38 functions the old window
  missed, so the denominator was 148 instead of 186.
- The doc block is found across attributes. Scanning only the lines directly
  above the `fn` missed a documented entry with an `#[inline]` between the doc
  comment and the item, such as
  `strided-basic/src/execution.rs:132` (`for_each_inner_block_preordered`).
  Before the fix the numerator was 138; both fixes together give 186 of 186.
