# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured on commit `ac8f744` of this branch over the maintained crates
(`strided-traits`, `strided-view`, `strided-basic`, `strided-kernel`,
`strided-fused`, `strided-nn`, `strided-perm`):

```bash
python3 scripts/count-audit-metrics.py .   # the seven source rows
cargo test --doc --workspace               # the doctest row, 123 passed, 0 ignored
```

`ac8f744` is a document-and-scripts commit; the Rust sources in it are the ones
this branch was rebased onto (`f7f9c5b`, #386). The counter only reads
`strided-*/**/*.rs`, so editing this file after that commit does not move any
row: a recount at a later commit of the same branch differs only if Rust source
changed.

The counter is a line-oriented approximation, not a Rust parser. It is accurate
enough for a trend, and it is not evidence that a file or a tree is clean: the
script's docstring lists what it still mismatches (`///` inside a string or a
block comment, strings inside attributes and macro invocations, braces inside
macro bodies, `unsafe extern`, macro-expanded `unsafe {`, `<`/`>` read as
brackets). Where a row matters, measure the thing itself — `cargo test --doc`
for the doctest row, a file read for a per-block `// SAFETY:` audit.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 52 | 0 |
| `// SAFETY:` comments | 302 | 29 |
| production `unsafe {` occurrences | 534 | 418 |
| public `Result`-returning `fn` | 186 | 172 |
| …of those with a `# Errors` section | 186 | 9 |
| runnable doctests | 123 | 5 |
| inline `#[cfg(test)] mod` blocks with a body | 1 | 53 |
| files over 1000 lines | 21 | 13 |

Three of the rows are already at or near their target because the work landed
between the issue's opening count and this baseline: the `# Errors` row reads
186/186 (the sweep of #381 left no public `Result`-returning `fn` without the
section — but the counter cannot see items a macro expands, so this is "no
item the scan can read", not an exhaustive claim), the doctests are the same
sweep, and the inline-test drop is the move
of the inline unit test modules into `*/tests/` directories (#362) — every
`src/` file now carries a `mod tests;` declaration, and the one remaining inline
block with a body is `strided-basic/benches/support/mod.rs:208`. The gap the
remediation issues should take next is `SAFETY` coverage, and 21 files are over
1000 lines. Read the old numbers as history and these as the baseline.

`SAFETY` is counted as two independent totals, not as a coverage ratio: 534
production `unsafe` occurrences and 302 `// SAFETY:` comments. A ratio would
claim that each of those comments documents one of those occurrences, and this
counter does not establish that pairing; the shared rule requires one comment
per block, so the per-block audit is the next measurement, not this one.

## Counting rules

`scripts/count-audit-metrics.py` is the only implementation of these rules, and
`scripts/test-count-audit-metrics.py` pins the cases a naive line scan gets
wrong:

```bash
python3 scripts/test-count-audit-metrics.py     # 8 tests
```

Production scope is `strided-*/src/**/*.rs` with a `tests/` path component and
inline `#[cfg(test)] mod NAME { .. }` bodies skipped for the first five rows;
the doc-comment, inline-module and file rows scan all of `strided-*/**/*.rs`
with a `target/` component excluded. Both exclusions compare repo-relative path
components. The scanner blanks block comments (nested), line comments, char
literals and string literals — including line-spanning `r#"…"#` — before
counting, which is the cheapest useful reading of `rules/rust/index.md:23` ("skip
`// SAFETY:` comments, doc comments, string literals, tests"); it is an
approximation with the limits listed in `scripts/count-audit-metrics.py`. The
doctest row is not in the counter: it is `cargo test --doc`, which is the
definition of "runnable"; the recorded value is the command's own output at this
tree (123 passed, 0 ignored).

| Row | Rule |
|-----|------|
| `// INVARIANT:` | `//` line comments (doc comments excluded, block comments excluded) containing the marker |
| `// SAFETY:` | the same for `// SAFETY:`; an independent total, not paired with the row below |
| `unsafe {` | occurrences of `unsafe` + `{` in the scanned text, comments and string literals blanked first, `unsafe` and `{` allowed on different lines; an independent total, not paired with the row above, and subject to the limits above |
| public `Result` `fn` | a `pub` (or `pub unsafe` / `pub const` / `pub extern "C"`) `fn` whose signature, read to the first `{` or `;`, has a `Result<` after its `->`; the line's own columns are preserved, so an `extern "C"` item is classified like any other |
| `# Errors` | the doc block nearest above such a `fn` contains a `/// # Errors` line; attribute lines (`#[inline]`, `#[must_use]`, `#[allow(..)]`, including multi-line ones) and blank lines between the doc block and the `fn` do not break the association |
| runnable doctests | `cargo test --doc` pass count at the repository root, not a fence count |
| inline test modules | `#[cfg(test)]` (optionally with further attributes) followed by `mod NAME { .. }`; a `mod NAME;` declaration counts 0, because its body lives in another file and moving the body is exactly what this row is meant to move |
| files over 1000 lines | `wc -l` over `strided-*/**/*.rs` |

### Counting defects found in review, and what each fix changed

Nine defects, all reproduced with a fixture before the fix and pinned by a test
after it. `scripts/test-count-audit-metrics.py` fails on the pre-fix counter for
every one of them except item 5 (`# Errors` across attributes), which an earlier
round had already fixed and which keeps its own test.

1. **The signature window.** The scan stopped after eight lines, so a `fn` whose
   return type sat past that window counted in neither the denominator nor the
   numerator (`strided-basic/src/copy_plan.rs:503`, `execute`, is thirteen lines).
   The signature is read to the first `{` or `;`: the denominator went 148 → 186.
2. **Attributes between the doc block and the item.** Scanning only the `///`
   lines above the `fn` missed a documented entry with `#[inline]` in between
   (`strided-basic/src/execution.rs:132`). Attribute and blank lines are stepped
   over: the numerator went 138 → 186.
3. **Comments and strings counted as code.** A file holding only
   `/* unsafe {} // SAFETY: x */` reported `unsafe 1, safety 1, invariant 1`, and
   the same text inside a line-spanning `r#"…"#` counted too. Block comments
   (which nest in Rust), line-spanning raw strings and char literals are now
   tracked across lines: the totals dropped 332 → 302 and 598 → 534, and the
   two rows are now stated as independent totals instead of a ratio. Both are
   still approximations — the remaining limits are listed in the script's
   docstring, not claimed away.
4. **Test-only files were counted as production.** The scope excluded a
   `tests/` path component but not a `tests.rs` file, the target of a
   `#[cfg(test)] mod tests;` declaration. Nineteen such files held 27 `unsafe`
   blocks and one `SAFETY` comment, and the totals were 561/303 until they were
   excluded. Counting braces over the string-blanked line, tracking an ordinary
   `"` string left open across lines, and reading the item's own return type
   instead of the first `->` in its signature (a callback parameter's) are the
   other three cases the counters got wrong; `scripts/test-count-audit-metrics.py`
   pins all four.
5. **`#[cfg(test)] mod tests;` swallowed the rest of the file.** The skip jumped
   to the next `{`, so production code after a declaration disappeared: the
   fixture's `pub fn retained() -> Result<(), ()> { unsafe {} … }` counted as
   `pub_result 0, unsafe 0`. A `NAME;` declaration now skips nothing.
6. **`pub extern "C" fn` fell out of the denominator.** Rebuilding the line from
   the scanned fragments moved `pub` to column 0 only for plain items, so
   `pub extern "C" fn retained() -> Result<(), ()>` counted as `pub_result 0`.
   The cleaned line keeps its own columns, so the item is classified like any
   other `pub fn`.
