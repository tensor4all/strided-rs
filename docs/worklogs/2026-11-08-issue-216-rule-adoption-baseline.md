# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured over the maintained crates (`strided-traits`, `strided-view`,
`strided-basic`, `strided-kernel`, `strided-fused`, `strided-nn`,
`strided-perm`) on the Rust sources **at this branch's head** — `c3b8999`, which
includes the merge of `main`; the branch was later rebased onto `8f89aae`
(#397, #398), which changed 11 more Rust files and left every row and the
doctest total unchanged — with the counter at the same head:

```bash
python3 scripts/count-audit-metrics.py .   # the seven source rows
cargo test --doc --workspace               # the doctest row, 127 passed, 0 ignored
```

```
invariant 59
safety 339
unsafe 537
pub_result 189 with_errors 189
inline_mods 1
files_over_1000 21
```

**A row is reproducible only with the pair (Rust sources, counter).** Both moved
since the first version of this table, which is why it now carries its own head:

- The counter changed: scanner fixes move the counts with no Rust change at all —
  the counter at `ac8f744` prints `safety 303, unsafe 561` on the sources that
  print `339, 537` here.
- The sources changed: the merge of `main` into this branch touched 36 Rust files,
  so the earlier `52 / 302 / 534 / 185 / 185 / 123` is a **historical** measurement
  at `f7f9c5b` (#386), not evidence for this tree. The rebase onto `8f89aae`
  (#397, #398) changed 11 further Rust files and the rows were re-run there:
  `59 / 339 / 537 / 189 / 189 / 127 / 1 / 21`, identical to the table below.

Record both when quoting a row: `git log -1 --format=%h -- scripts/count-audit-metrics.py`
for the counter and `git log -1 --format=%h` for the sources.

The counter is a line-oriented approximation, not a Rust parser. It is accurate
enough for a trend, and it is not evidence that a file or a tree is clean: the
script's docstring lists what it still mismatches (a string inside an attribute,
braces inside macro bodies, `unsafe extern`, macro-expanded `unsafe {`, `<`/`>`
read as brackets, a macro-produced declaration). Where a row matters, measure the thing itself — `cargo test --doc`
for the doctest row, a file read for a per-block `// SAFETY:` audit.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 59 | 0 |
| `// SAFETY:` comments | 339 | 29 |
| production `unsafe {` occurrences | 537 | 418 |
| public `Result`-returning `fn` | 189 | 172 |
| …of those with a `# Errors` section | 189 | 9 |
| runnable doctests | 127 | 5 |
| inline `#[cfg(test)] mod` blocks with a body | 1 | 53 |
| files over 1000 lines | 21 | 13 |

Three of the rows are already at or near their target because the work landed
between the issue's opening count and this baseline: the `# Errors` row reads
189/189 (the sweep of #381 left no public `Result`-returning `fn` without the
section — but a macro invocation is not expanded (see Known limits), so this is
"no item the scan can read", not an exhaustive claim), the doctests are the same
sweep, and the inline-test drop is the move
of the inline unit test modules into `*/tests/` directories (#362) — the files
that have unit tests carry a test-module declaration (some name it something other
than `tests`, so do not read the absence of `mod tests;` as the absence of unit
tests), and the one remaining inline
block with a body is `strided-basic/benches/support/mod.rs:400`. The gap the
remediation issues should take next is `SAFETY` coverage, and 21 files are over
1000 lines. Read the old numbers as history and these as the baseline.

`SAFETY` is counted as two independent totals, not as a coverage ratio: 537
production `unsafe` occurrences and 339 `// SAFETY:` comments. A ratio would
claim that each of those comments documents one of those occurrences, and this
counter does not establish that pairing; the shared rule requires one comment
per block, so the per-block audit is the next measurement, not this one.

## Counting rules

`scripts/count-audit-metrics.py` is the only implementation of these rules, and
`scripts/test-count-audit-metrics.py` pins the cases a naive line scan gets
wrong:

```bash
python3 scripts/test-count-audit-metrics.py     # prints its own test count
```

Production scope is `strided-*/src/**/*.rs` with a `tests/` path component, a
`tests.rs` file, and the module subtree of every test-only declaration excluded —
`#[cfg(test)] mod NAME;` excludes `NAME.rs`, `NAME/mod.rs` and everything below
`NAME/`, an inline `#[cfg(test)] mod NAME { .. }` excludes `NAME/` as well, and a
`#[path = "..."]` attribute names its file directly. Ordinary declarations resolve
in the declaring module's directory (`foo/` for `foo.rs`, the containing directory
for `lib.rs`/`mod.rs`); an explicit path resolves like rustc's: against the
declaring file's own directory at the top level, and against the enclosing module's
directory when it sits inside an inline module, so it may point outside the module
subtree. A file that both a test-only and a production declaration load stays in
scope: test-only is a greatest fixed point over every declaration, including ones
that are themselves excluded. Inline `#[cfg(test)] mod NAME { .. }` bodies are skipped
for the first five rows; the `# Errors` row uses the same production files as the
`Result` row, and the inline-module and file rows scan all of `strided-*/**/*.rs`
with a `target/` component excluded.
Both exclusions compare repo-relative path components. The scanner blanks block
comments (nested), line comments, char literals (including escaped ones such as
`'\"'` and `'\u{1F600}'`) and string literals — including line-spanning
`r#"…"#` and ordinary `"…"` — before counting, which is the cheapest useful
reading of `rules/rust/index.md:23` ("skip `// SAFETY:` comments, doc comments,
string literals, tests"); it is an approximation with the limits listed in
`scripts/count-audit-metrics.py`. The doctest row is not in the counter: it is
`cargo test --doc`, which is the definition of "runnable"; the recorded value is
the command's own output at this tree (127 passed, 0 ignored).

| Row | Rule |
|-----|------|
| `// INVARIANT:` | `//` line comments (doc comments excluded, block comments excluded) containing the marker |
| `// SAFETY:` | the same for `// SAFETY:`; an independent total, not paired with the row below |
| `unsafe {` | occurrences of `unsafe` + `{` in the scanned text, comments and string literals blanked first, `unsafe` and `{` allowed on different lines; an independent total, not paired with the row above, and subject to the limits above |
| public `Result` `fn` | a `pub` (or `pub unsafe` / `pub const` / `pub extern "C"`) `fn` whose signature, read to the first `{` or `;`, has a `Result<` after its own `->`; the line's own columns are preserved, so an `extern "C"` item is classified like any other; an arrow is consumed as one token, so a callback parameter's `-> Result<…>` is never read as the item's return type |
| `# Errors` | the doc block nearest above such a `fn` contains a `/// # Errors` heading — the word boundary is required, so `/// # ErrorsAreNotDocumented` does not count; attribute lines (`#[inline]`, `#[must_use]`, `#[allow(..)]`, including multi-line ones) and blank lines between the doc block and the `fn` do not break the association |
| runnable doctests | `cargo test --doc` pass count at the repository root, not a fence count |
| inline test modules | `#[cfg(test)]` (optionally with further attributes) followed by `mod NAME { .. }`; a `mod NAME;` declaration counts 0, because its body lives in another file and moving the body is exactly what this row is meant to move |
| files over 1000 lines | logical lines (`str.splitlines()`) over `strided-*/**/*.rs`, which is one more than `wc -l` for a file without a final newline |

### Known limits: the counts are a lower bound

The counter is not a Rust parser. It blanks block comments (nested), line
comments, char literals and string literals, and is line-oriented everywhere
else. These cases are known to be miscounted, and they are **not** fixed here:

- A `macro_rules!` template is scanned where it is written: a template spelling
  `pub fn $name(..) -> Result<..>` counts once (`strided-nn/src/softmax.rs:179`),
  and its invocations are not expanded, so an item only the expansion creates is
  never seen.
- Braces inside a macro body that the scanner reads as code still move the
  `#[cfg(test)] mod` skip and the inline-module count.
- `unsafe` is counted as the text `unsafe` + `{`, so `unsafe extern` blocks, an
  `unsafe fn` whose brace belongs to another item, and a macro that expands to
  `unsafe {` are not distinguished.
- A `<` or `>` used as a comparison operator inside a signature can misplace the
  signature's end; arrows are handled, comparison operators are not.
- A `mod NAME;` whose declaration is itself produced by a macro is not followed.

Consequences for reading a row:

- The `// INVARIANT:`, `// SAFETY:` and `unsafe {` rows count text the scan
  believes is code, and text it misclassifies is dropped rather than invented, so
  they are a **lower bound** for the source text in scope: the true count is at
  least the printed one.
- The `Result` rows are a lower bound in the same way, except that the
  comparison-operator limit above can move an item either way.
- This direction was falsified once: a `///` line inside a block comment made all
  three rows over-count (item 11). Treat a future over-count as a counter bug to
  fix, not as a licence to read a row as exact.
- A zero means "the scan found none", not "none exists". Where a row matters,
  measure the thing itself: `cargo test --doc` for the doctest row, a per-block
  read for a `// SAFETY:` audit.

### The scanners are pinned by tests, not by review wording

Every definition this baseline depends on is exercised by
`scripts/test-count-audit-metrics.py` (44 tests, run by the `scripts` CI job): the signature window,
attribute rows between a doc block and its item, comments and strings blanked before counting, test-only
files excluded through the module graph, `#[path]` resolution that follows rustc's, and `# Errors` matched
as a heading with a word boundary. A definition that regresses fails a fixture there, so the numbers above
mean the same thing when a later change re-measures them. They remain a lower bound either way, for the
reasons in "Known limits".
