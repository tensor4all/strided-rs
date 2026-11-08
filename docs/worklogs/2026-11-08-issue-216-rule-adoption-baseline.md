# strided-rs#216 — rule adoption baseline

## Scope

[#216](https://github.com/tensor4all/strided-rs/issues/216) moves the
repository-neutral rules out of `REPOSITORY_RULES.md` and into references to
`tensor4all/tensor4all-agent-rules`. The second half of the issue is the audit
baseline: the gap counts the rule-remediation issues check themselves against.

Measured on this branch's tree @ `06e1219` over the maintained crates
(`strided-traits`, `strided-view`, `strided-basic`, `strided-kernel`,
`strided-fused`, `strided-nn`, `strided-perm`):

```bash
python3 scripts/count-audit-metrics.py .
cargo test --doc --workspace        # the doctest row
```

These are source counts: they depend on the tree, not on the machine. `06e1219`
is rebased over `f7f9c5b` (#386, `fix(view): stop caching the pointer that
`data_mut` invalidates`), so this tree contains a Rust change that the first
draft of this table did not. For the `unsafe` / `SAFETY` rows that change is not
what moved the numbers: on the pre-#386 tree (`bf07ff1`) the previous counter
reports 558 and 300, the current one reports 561 and 303, and this tree reports
the same 561 and 303. The difference is the counter's comment handling, not
#386. The earlier draft's 556 came from the `41cb04d` tree with that previous
counter and is superseded by the row above; every number in this file is
re-measured at `06e1219` with `scripts/count-audit-metrics.py`.

| Metric | Count | #216's opening count |
|--------|-------|----------------------|
| `// INVARIANT:` markers | 52 | 0 |
| `// SAFETY:` comments | 303 | 29 |
| production `unsafe {` occurrences | 561 | 418 |
| public `Result`-returning `fn` | 186 | 172 |
| …of those with a `# Errors` section | 186 | 9 |
| runnable doctests | 123 | 5 |
| inline `#[cfg(test)] mod` blocks with a body | 1 | 53 |
| files over 1000 lines | 21 | 13 |

Three of the rows are already at or near their target because the work landed
between the issue's opening count and this baseline: the `# Errors` row is
closed (the sweep of #381 left no public `Result`-returning `fn` without the
section), the doctests are the same sweep, and the inline-test drop is the move
of the inline unit test modules into `*/tests/` directories (#362) — every
`src/` file now carries a `mod tests;` declaration, and the one remaining inline
block with a body is `strided-basic/benches/support/mod.rs:208`. The gap the
remediation issues should take next is `SAFETY` coverage, and 21 files are over
1000 lines. Read the old numbers as history and these as the baseline.

`SAFETY` is counted as two independent totals, not as a coverage ratio: 561
production `unsafe` occurrences and 303 `// SAFETY:` comments. A ratio would
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

Production scope is `strided-*/src/**/*.rs` with `*/tests/` excluded and inline
`#[cfg(test)] mod NAME { .. }` bodies skipped for the first five rows; the
doc-comment, inline-module and file rows scan all of `strided-*/**/*.rs`
(`target/` excluded). Comments, doc comments and string literals are removed
from the source text before anything is counted, because `rules/rust/index.md:23`
counts production `unsafe` "precisely (skip `// SAFETY:` comments, doc comments,
string literals, tests, generated code)". The doctest row is not in the counter:
it is `cargo test --doc`, which is the definition of "runnable"; the recorded
value is the command's own output at this tree (123 passed, 0 ignored).

| Row | Rule |
|-----|------|
| `// INVARIANT:` | `//` line comments (doc comments excluded, block comments excluded) containing the marker |
| `// SAFETY:` | the same for `// SAFETY:`; an independent total, not paired with the row below |
| `unsafe {` | occurrences of `unsafe` + `{` in code, not blocks, after block comments, line comments, doc comments, char literals and string literals (including line-spanning `r#"…"#`) are removed; an independent total, not paired with the row above |
| public `Result` `fn` | a `pub` (or `pub unsafe` / `pub const` / `pub extern "C"`) `fn` whose signature, read to the first `{` or `;`, has a `Result<` after its `->`; the line's own columns are preserved, so an `extern "C"` item is classified like any other |
| `# Errors` | the doc block nearest above such a `fn` contains a `/// # Errors` line; attribute lines (`#[inline]`, `#[must_use]`, `#[allow(..)]`, including multi-line ones) and blank lines between the doc block and the `fn` do not break the association |
| runnable doctests | `cargo test --doc` pass count at the repository root, not a fence count |
| inline test modules | `#[cfg(test)]` (optionally with further attributes) followed by `mod NAME { .. }`; a `mod NAME;` declaration counts 0, because its body lives in another file and moving the body is exactly what this row is meant to move |
| files over 1000 lines | `wc -l` over `strided-*/**/*.rs` |

### Counting defects found in review, and what each fix changed

Five defects, all reproduced with a fixture before the fix and pinned by a test
after it. `scripts/test-count-audit-metrics.py` fails on the pre-fix counter for
the first four; the fifth (`# Errors` across attributes) was already fixed by an
earlier round and has its own test.

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
   tracked across lines: the totals dropped 332 → 303 and 598 → 561 before the
   tree moved, and the two rows are now stated as independent totals instead of
   a ratio.
4. **`#[cfg(test)] mod tests;` swallowed the rest of the file.** The skip jumped
   to the next `{`, so production code after a declaration disappeared: the
   fixture's `pub fn retained() -> Result<(), ()> { unsafe {} … }` counted as
   `pub_result 0, unsafe 0`. A `NAME;` declaration now skips nothing.
5. **`pub extern "C" fn` fell out of the denominator.** Rebuilding the line from
   the scanned fragments moved `pub` to column 0 only for plain items, so
   `pub extern "C" fn retained() -> Result<(), ()>` counted as `pub_result 0`.
   The cleaned line keeps its own columns, so the item is classified like any
   other `pub fn`.
