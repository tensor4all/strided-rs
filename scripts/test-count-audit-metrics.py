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
            (src / name).write_text(body, encoding="utf-8")
        return mod.count(pathlib.Path(tmp))


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
