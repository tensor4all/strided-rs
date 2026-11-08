#!/usr/bin/env python3
"""Self-contained tests for scripts/count-audit-metrics.py.

Run with `python3 scripts/test-count-audit-metrics.py`; no pytest needed.

Each test writes a small `strided-x/src/*.rs` fixture tree and asserts what the
counter reports for it. The cases are the ones a naive line scan gets wrong:
block comments and line-spanning raw strings counted as code, a
`#[cfg(test)] mod tests;` declaration swallowing the production code after it,
and `pub extern "C" fn` dropped from the `Result` denominator.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "count-audit-metrics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("count_audit_metrics", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def count_sources(files: dict[str, str]) -> dict[str, int]:
    """Run the counter over a throwaway `strided-x/src/` tree built from `files`."""
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "strided-x" / "src"
        src.mkdir(parents=True)
        for name, body in files.items():
            target = src / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        return mod.count(pathlib.Path(tmp))


def char_literal_is_blanked(literal: str) -> bool:
    """True when the scanner blanks `literal` out of the code rows."""
    mod = load_module()
    rows = mod.scan_file(f"const C: char = {literal};\n")
    return literal not in "".join(str(row["code_no_strings"]) for row in rows)


def test_block_comment_is_not_code() -> None:
    """A block comment holds no production `unsafe`, `// SAFETY:` or `// INVARIANT:`."""
    counts = count_sources({"only_comment.rs": "/* unsafe {} // SAFETY: x // INVARIANT: y */\n"})
    assert counts["unsafe"] == 0, counts
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_line_spanning_raw_string_is_not_code() -> None:
    counts = count_sources(
        {"note.rs": 'const NOTE: &str = r#"\nunsafe {} // SAFETY: x // INVARIANT: y\n"#;\n'}
    )
    assert counts["unsafe"] == 0, counts
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_code_after_a_test_module_declaration_is_counted() -> None:
    """`#[cfg(test)] mod tests;` moves a body out of the file; it skips nothing."""
    counts = count_sources(
        {
            "decl.rs": (
                "#[cfg(test)]\n"
                "mod tests;\n"
                "\n"
                "pub fn retained() -> Result<(), ()> {\n"
                "    unsafe {}\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["unsafe"] == 1, counts
    assert counts["inline_mods"] == 0, counts


def test_inline_test_body_is_skipped_and_counted() -> None:
    counts = count_sources(
        {
            "body.rs": (
                "#[cfg(test)]\n"
                "mod tests {\n"
                "    fn inner() {\n"
                "        unsafe {}\n"
                "    }\n"
                "}\n"
                "\n"
                "pub fn live() -> Result<(), ()> {\n"
                "    unsafe {}\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["inline_mods"] == 1, counts
    assert counts["unsafe"] == 1, counts
    assert counts["pub_result"] == 1, counts


def test_extern_c_function_is_in_the_result_denominator() -> None:
    counts = count_sources(
        {"extern_c.rs": 'pub extern "C" fn retained() -> Result<(), ()> {\n    Ok(())\n}\n'}
    )
    assert counts["pub_result"] == 1, counts


def test_unsafe_split_across_lines_is_counted() -> None:
    """`unsafe` and its `{` may sit on different lines."""
    counts = count_sources(
        {
            "split.rs": (
                "pub fn f() {\n"
                "    unsafe\n"
                "    {\n"
                "        core::hint::black_box(1);\n"
                "    }\n"
                "}\n"
            )
        }
    )
    assert counts["unsafe"] == 1, counts


def test_array_semicolon_does_not_end_the_signature() -> None:
    """`[u8; 4]` in a parameter list is not the end of the item."""
    counts = count_sources(
        {
            "array_param.rs": (
                "pub fn f(buf: [u8; 4]) -> Result<(), ()> {\n"
                "    let _ = buf;\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts


def test_blank_line_between_cfg_test_and_mod_is_an_inline_block() -> None:
    counts = count_sources(
        {
            "blanks.rs": (
                "#[cfg(test)]\n"
                "\n"
                "mod tests {\n"
                "    fn inner() {\n"
                "        unsafe {}\n"
                "    }\n"
                "}\n"
            )
        }
    )
    assert counts["inline_mods"] == 1, counts
    assert counts["unsafe"] == 0, counts


def test_path_exclusions_use_repo_relative_components() -> None:
    """A checkout under a directory named `target` or `tests` still counts its files."""
    mod = load_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "target" / "tests" / "checkout"
        src = root / "strided-x" / "src"
        src.mkdir(parents=True)
        (src / "real.rs").write_text("pub fn f() {\n    unsafe {}\n}\n", encoding="utf-8")
        counts = mod.count(root)
    assert counts["unsafe"] == 1, counts


def test_doc_block_survives_attributes() -> None:
    """`# Errors` above `#[inline]` / `#[must_use]` still documents the item."""
    counts = count_sources(
        {
            "documented.rs": (
                "/// # Errors\n"
                "///\n"
                "/// Never returns an error.\n"
                "#[inline]\n"
                "#[must_use]\n"
                "pub fn documented() -> Result<(), ()> {\n"
                "    Ok(())\n"
                "}\n"
            ),
            "undocumented.rs": "pub fn undocumented() -> Result<(), ()> {\n    Ok(())\n}\n",
        }
    )
    assert counts["pub_result"] == 2, counts
    assert counts["with_errors"] == 1, counts


def test_multiline_signature_is_in_the_result_denominator() -> None:
    """The signature is read to the first `{` or `;`, not to a fixed window."""
    counts = count_sources(
        {
            "long_signature.rs": (
                "pub fn long<T>(\n"
                "    a: usize,\n"
                "    b: usize,\n"
                "    c: usize,\n"
                "    d: usize,\n"
                "    e: usize,\n"
                "    f: usize,\n"
                "    g: usize,\n"
                ") -> Result<(), ()> {\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts


def test_real_unsafe_blocks_and_comments_are_counted() -> None:
    """Positive control: the counter is not vacuous."""
    counts = count_sources(
        {
            "real.rs": (
                "// SAFETY: the caller proved the bound.\n"
                "// INVARIANT: the pair stays ordered.\n"
                "pub fn f() {\n"
                "    // SAFETY: the index is in bounds.\n"
                "    unsafe {}\n"
                "}\n"
            )
        }
    )
    assert counts["unsafe"] == 1, counts
    assert counts["safety"] == 2, counts
    assert counts["invariant"] == 1, counts


def test_a_tests_rs_sibling_of_a_mod_declaration_is_test_only() -> None:
    """`#[cfg(test)] mod tests;` moves the body to `tests.rs`; it is not production."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod tests;\n\npub fn live() {\n    unsafe {}\n}\n",
            "tests.rs": "#[test]\nfn test_only() {\n    unsafe {}\n}\n",
        }
    )
    assert counts["unsafe"] == 1, counts


def test_braces_inside_a_test_string_do_not_swallow_production() -> None:
    counts = count_sources(
        {
            "fixture.rs": (
                "#[cfg(test)]\n"
                "mod tests {\n"
                '    const S: &str = "{";\n'
                "}\n"
                "pub fn live() -> Result<(), ()> {\n"
                "    unsafe {}\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["unsafe"] == 1, counts


def test_line_spanning_ordinary_string_is_not_code() -> None:
    counts = count_sources(
        {"fixture.rs": 'const NOTE: &str = "\nunsafe {} // SAFETY: x // INVARIANT: y\n";\n'}
    )
    assert counts["unsafe"] == 0, counts
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_a_comment_shaped_line_inside_an_open_string_is_not_a_comment() -> None:
    """A `//` or `///` line that begins inside a `"…"` is string content, not a comment or doc line."""
    counts = count_sources(
        {
            "fixture.rs": (
                'const NOTE: &str = "\n'
                "// SAFETY: x\n"
                "// INVARIANT: y\n"
                '";\n'
                "pub fn live() {\n    unsafe {}\n}\n"
            )
        }
    )
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts
    assert counts["unsafe"] == 1, counts


def test_a_callback_return_type_is_not_the_items_return_type() -> None:
    counts = count_sources(
        {"fixture.rs": "pub fn accepts(f: fn() -> Result<(), ()>) {\n    let _ = f;\n}\n"}
    )
    assert counts["pub_result"] == 0, counts
    counts = count_sources({"fixture.rs": "pub fn returns() -> Result<(), ()> {\n    Ok(())\n}\n"})
    assert counts["pub_result"] == 1, counts


def test_two_callback_arrows_do_not_move_the_items_return_type() -> None:
    """Two callback parameters: the first arrow's `>` must not close the parameter list."""
    counts = count_sources(
        {
            "fixture.rs": (
                "pub fn returns(f: fn() -> (), g: fn() -> ()) -> Result<(), ()> {\n"
                "    let _ = (f, g);\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    counts = count_sources(
        {
            "fixture.rs": (
                "pub fn accepts(f: fn() -> (), g: fn() -> Result<(), ()>) {\n"
                "    let _ = (f, g);\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 0, counts


def test_a_differently_named_test_only_module_is_not_production() -> None:
    """`#[cfg(test)] mod helpers;` puts `helpers.rs` out of the production scope."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod helpers;\n",
            "helpers.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_test_only_module_directory_is_not_production() -> None:
    """The target may be `NAME/mod.rs`, and a dropped file's own test-only mods follow."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod helpers;\n",
            "helpers/mod.rs": "#[cfg(test)]\nmod deep;\n",
            "helpers/deep.rs": "pub fn deep() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_an_escaped_char_literal_does_not_open_a_string() -> None:
    r"""`'\"'`, `'\''`, `'\\'` and `'\u{1F600}'` are char literals, blanked like any string."""
    for literal in (r"'\"'", r"'\''", r"'\\'", r"'\u{1F600}'", "'a'", "b'\"'"):
        assert char_literal_is_blanked(literal), literal
        counts = count_sources(
            {
                "fixture.rs": (
                    f"const C: char = {literal};\n"
                    "pub fn live() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n"
                )
            }
        )
        assert counts["unsafe"] == 1, (literal, counts)
        assert counts["pub_result"] == 1, (literal, counts)


def test_a_module_declaration_resolves_in_its_own_file() -> None:
    """`mod helpers;` in `foo.rs` means `foo/helpers.rs`, not the sibling `helpers.rs`."""
    counts = count_sources(
        {
            "lib.rs": "mod foo;\nmod helpers;\n",
            "foo.rs": "#[cfg(test)]\nmod helpers;\n",
            "foo/helpers.rs": "\n",
            "helpers.rs": "pub fn live() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 1, counts
    assert counts["pub_result"] == 1, counts


def test_test_only_exclusion_covers_the_module_subtree() -> None:
    """An ordinary `mod deep;` below a test-only module is test-only too."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod helpers;\n",
            "helpers/mod.rs": "mod deep;\n",
            "helpers/deep.rs": "pub fn deep() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_where_clause_callback_is_not_the_items_return_type() -> None:
    """A bound's arrow sits past a top-level `where`, not in the item's return type."""
    counts = count_sources(
        {
            "fixture.rs": (
                "pub fn accepts<F>(f: F) where F: Fn() -> Result<(), ()> {\n"
                "    let _ = f;\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 0, counts
    counts = count_sources(
        {
            "fixture.rs": (
                "pub fn returns<F>(f: F) -> Result<(), ()> where F: Fn() -> () {\n"
                "    let _ = f;\n"
                "    Ok(())\n"
                "}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts


def test_a_doc_shaped_line_inside_a_block_comment_is_not_a_doc_line() -> None:
    """`///` inside `/* */` is block-comment text, and the terminator on that line counts."""
    counts = count_sources(
        {
            "fixture.rs": (
                "/*\n"
                "/// */\n"
                'const S: &str = "*/ unsafe {} // SAFETY: x // INVARIANT: y";\n'
            )
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_a_raw_string_continuation_is_not_code() -> None:
    """A raw string's closing line is string content, so it cannot start a `pub fn`."""
    counts = count_sources(
        {"fixture.rs": 'const S: &str = r#"\npub fn ghost() -> Result<(), ()> {}"#;\n'}
    )
    assert counts["pub_result"] == 0, counts
    counts = count_sources(
        {"fixture.rs": 'const S: &str = r#"\npub fn ghost() { unsafe {} }"#;\n'}
    )
    assert counts["unsafe"] == 0, counts


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"count-audit-metrics: {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
