# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured over the maintained crates (`strided-traits`, `strided-view`,
`strided-basic`, `strided-kernel`, `strided-fused`, `strided-nn`,
`strided-perm`) on the Rust sources this branch was rebased onto (`f7f9c5b`,
#386) — unchanged from `ac8f744` through this head, `git diff ac8f744 HEAD --
'strided-*/**/*.rs'` is empty — **with the counter at this branch's head**:

```bash
python3 scripts/count-audit-metrics.py .   # the seven source rows
cargo test --doc --workspace               # the doctest row, 123 passed, 0 ignored
```

```
invariant 52
safety 302
unsafe 534
pub_result 185 with_errors 185
inline_mods 1
files_over_1000 21
```

**The counter's revision is part of the measurement.** A row is only reproducible
with the pair (Rust sources, counter), because scanner fixes move the counts
without any Rust change: the counter at `ac8f744` prints `safety 303, unsafe 561`
on these same sources, and the earlier table in this file quoted those values
under the source revision alone. Record both, e.g.
`git log -1 --format=%h -- scripts/count-audit-metrics.py` for the counter and
`git diff ac8f744 HEAD -- 'strided-*/**/*.rs'` to show the sources did not move.

The counter is a line-oriented approximation, not a Rust parser. It is accurate
enough for a trend, and it is not evidence that a file or a tree is clean: the
script's docstring lists what it still mismatches (strings inside attributes and
macro invocations, braces inside macro bodies, `unsafe extern`, macro-expanded
`unsafe {`, `<`/`>` read as brackets, a macro-produced declaration). Where a row matters, measure the thing itself — `cargo test --doc`
for the doctest row, a file read for a per-block `// SAFETY:` audit.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 52 | 0 |
| `// SAFETY:` comments | 302 | 29 |
| production `unsafe {` occurrences | 534 | 418 |
| public `Result`-returning `fn` | 185 | 172 |
| …of those with a `# Errors` section | 185 | 9 |
| runnable doctests | 123 | 5 |
| inline `#[cfg(test)] mod` blocks with a body | 1 | 53 |
| files over 1000 lines | 21 | 13 |

Three of the rows are already at or near their target because the work landed
between the issue's opening count and this baseline: the `# Errors` row reads
185/185 (the sweep of #381 left no public `Result`-returning `fn` without the
section — but a macro invocation is not expanded (see Known limits), so this is
"no item the scan can read", not an exhaustive claim), the doctests are the same
sweep, and the inline-test drop is the move
of the inline unit test modules into `*/tests/` directories (#362) — the files
that have unit tests carry a test-module declaration (some name it something other
than `tests`, so do not read the absence of `mod tests;` as the absence of unit
tests), and the one remaining inline
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
python3 scripts/test-count-audit-metrics.py     # prints its own test count
```

Production scope is `strided-*/src/**/*.rs` with a `tests/` path component, a
`tests.rs` file, and the module subtree of every test-only declaration excluded —
`#[cfg(test)] mod NAME;` excludes `NAME.rs`, `NAME/mod.rs` and everything below
`NAME/`, an inline `#[cfg(test)] mod NAME { .. }` excludes `NAME/` as well, and a
`#[path = "..."]` attribute names its file directly. Ordinary declarations resolve
in the declaring module's directory (`foo/` for `foo.rs`, the containing directory
for `lib.rs`/`mod.rs`); an explicit path resolves against the declaring file's own
directory, so it may point outside the module subtree. The exclusion set is built
from every candidate file, including ones that are themselves excluded, which is
why one pass is enough. Inline `#[cfg(test)] mod NAME { .. }` bodies are skipped
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
the command's own output at this tree (123 passed, 0 ignored).

| Row | Rule |
|-----|------|
| `// INVARIANT:` | `//` line comments (doc comments excluded, block comments excluded) containing the marker |
| `// SAFETY:` | the same for `// SAFETY:`; an independent total, not paired with the row below |
| `unsafe {` | occurrences of `unsafe` + `{` in the scanned text, comments and string literals blanked first, `unsafe` and `{` allowed on different lines; an independent total, not paired with the row above, and subject to the limits above |
| public `Result` `fn` | a `pub` (or `pub unsafe` / `pub const` / `pub extern "C"`) `fn` whose signature, read to the first `{` or `;`, has a `Result<` after its own `->`; the line's own columns are preserved, so an `extern "C"` item is classified like any other; an arrow is consumed as one token, so a callback parameter's `-> Result<…>` is never read as the item's return type |
| `# Errors` | the doc block nearest above such a `fn` contains a `/// # Errors` line; attribute lines (`#[inline]`, `#[must_use]`, `#[allow(..)]`, including multi-line ones) and blank lines between the doc block and the `fn` do not break the association |
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

### Counting defects found in review, and what each fix changed

Fifteen findings. Items 1-9 and 11-15 are scanner defects, each reproduced with
a fixture before the fix and pinned by a test after it; the suite fails on the
pre-fix counter for every one of them except item 2 (`# Errors` across
attributes), which an earlier round had already fixed and which keeps its own
test. Items 1-6 came from the first review round, 7-10 from the second
(`2026-11-08T05:25Z`, head `96dc462`) and 11-15 from the third (head `274372d`).
Item 10 is an attribution error rather than a scanner defect, so it has no
regression test.

1. **The signature window.** The scan stopped after eight lines, so a `fn` whose
   return type sat past that window counted in neither the denominator nor the
   numerator (`strided-basic/src/copy_plan.rs:503`, `execute`, is thirteen lines).
   The signature is read to the first `{` or `;`: the denominator went 148 → 186.
2. **Attributes between the doc block and the item.** Scanning only the `///`
   lines above the `fn` missed a documented entry with `#[inline]` in between
   (`strided-basic/src/execution.rs:132`). Attribute and blank lines are stepped
   over: the numerator went 138 → 186.
3. **Comments and strings counted as code.** A file holding only
   `/* unsafe {} // SAFETY: x // INVARIANT: y */` reported
   `unsafe 1, safety 1, invariant 1`, and the same text inside a line-spanning
   `r#"…"#` counted too. Block comments
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
   `"` string left open across lines — including a `//` or `///` line that begins
   inside it, which was still read as a comment or a doc line — and reading the
   item's own return type instead of the first `->` in its signature (a callback
   parameter's) are the other three cases the counters got wrong;
   `scripts/test-count-audit-metrics.py` pins all four.
5. **`#[cfg(test)] mod tests;` swallowed the rest of the file.** The skip jumped
   to the next `{`, so production code after a declaration disappeared: the
   fixture's `pub fn retained() -> Result<(), ()> { unsafe {} … }` counted as
   `pub_result 0, unsafe 0`. A `NAME;` declaration now skips nothing.
6. **`pub extern "C" fn` fell out of the denominator.** Rebuilding the line from
   the scanned fragments moved `pub` to column 0 only for plain items, so
   `pub extern "C" fn retained() -> Result<(), ()>` counted as `pub_result 0`.
   The cleaned line keeps its own columns, so the item is classified like any
   other `pub fn`.
7. **A test-only module with any name stayed production.** The scope dropped a
   file named `tests.rs` but did not read the declaration that makes a file
   test-only: `strided-perm/src/lib.rs`'s `#[cfg(test)] mod block;`, `mod kernel;`
   and `mod order;` kept `strided-perm/src/{block,kernel,order}.rs` in the totals.
   The fixture `#[cfg(test)] mod helpers;` plus `helpers.rs` reported
   `unsafe 1, pub_result 1` where production holds neither. The declaration is now
   followed (`NAME.rs` and `NAME/mod.rs`; item 15 replaces the fixed-point pass
   with subtree exclusion): three files left the scope and the `Result`
   denominator went 186 → 185. No `unsafe` or `SAFETY` occurrence moved, so those
   rows stay 534/302.
8. **Two callback arrows moved the item's return type.** `read_signature` and
   `returns_result` counted the `>` of `->` as a closing bracket, so the first
   callback's arrow closed the parameter list and the second one stood where the
   item's own arrow belongs. `pub fn accepts(f: fn() -> (), g: fn() -> Result<(), ()>)`
   counted as `pub_result 1` (expected 0) and
   `pub fn returns(f: fn() -> (), g: fn() -> ()) -> Result<(), ()>` as `0`
   (expected 1). Both fixtures compile with `rustc`; an arrow is now consumed as
   one token in both functions.
9. **An escaped character literal opened a fictitious string.** The scanner only
   recognized `'x'`, so the double quote in `const QUOTE: char = '\"';` started a
   string and swallowed the rest of the file: a following
   `pub fn live() -> Result<(), ()> { unsafe {} Ok(()) }` counted as
   `unsafe 0, pub_result 0` instead of `1, 1`. Char literals are now matched as
   `'(\\.|[^\\'])'` together with the `\u{…}` form, which covers `'\"'`,
   `'\''` and `'\u{1F600}'`.
10. **The table quoted the counter's numbers under the source revision alone.**
    The baseline above said "measured on `ac8f744`" and that a later recount
    differs only when Rust changes, but the counter itself changed afterwards: the
    counter at `ac8f744` prints `safety 303, unsafe 561` on the very sources that
    print `302, 534` here. Both revisions are now recorded, and the earlier
    `561/303` under `ac8f744` is what finding 4 of the previous round had already
    removed.
11. **A `///` line inside a block comment was taken as a doc line.** The doc fast
    path skipped the line without updating the scanner state, so the real `*/` on
    that line was lost and the following
    `const S: &str = "*/ unsafe {} // SAFETY: x // INVARIANT: y";` was read as code:
    `unsafe 1, safety 1, invariant 1` instead of `0, 0, 0`. This is the one case
    found so far where the rows *over*-count. The doc path now requires no open
    block comment, raw string or ordinary string, as the ordinary-comment path
    already did.
12. **A raw string's closing line was code.** The continuation span up to `"#` was
    kept in the comment-preserving field that `FUNCTION_RE` matches, so
    `const S: &str = r#"` on one line and `pub fn ghost() -> Result<(), ()> {}"#;`
    on the next counted a public `Result` function: `pub_result 1`, expected 0. The
    span is now blanked in both fields, as the ordinary-string continuation already
    was.
13. **A `where`-clause bound's arrow became the item's return type.**
    `pub fn accepts<F>(f: F) where F: Fn() -> Result<(), ()>` counted as
    `pub_result 1`, because the bound's arrow sits at bracket depth zero once the
    parameter list closes. A top-level `where` now ends the search for the item's
    own arrow.
14. **A test-only module's declaration resolved against the wrong directory.**
    `mod helpers;` in `strided-x/src/foo.rs` was resolved to `src/helpers.rs`
    instead of `src/foo/helpers.rs`, so a production `src/helpers.rs` could be
    dropped: the fixture reported `unsafe 0, pub_result 0` where production held
    one of each. Declarations now resolve in the declaring module's directory
    (`foo/` for `foo.rs`, the containing directory for `lib.rs`/`mod.rs`).
15. **A test-only module's subtree was not excluded.** Only the declared file was
    dropped, so an ordinary `mod deep;` inside a test-only `helpers/mod.rs` kept
    `helpers/deep.rs` in the totals (`unsafe 1, pub_result 1`, expected 0). The
    whole subtree (`NAME.rs`, `NAME/mod.rs`, `NAME/**`) is excluded now, which also
    makes the old repeated-rescan loop unnecessary.
