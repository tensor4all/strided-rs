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


def test_scan_line_keeps_the_column_of_every_character() -> None:
    """Both returned fields keep the line's length, including a `*/` terminator."""
    mod = load_module()
    raw = "/*x*/unsafe {}"
    code, no_strings, _ = mod.scan_line(
        raw, {"block_comment": 0, "raw_string": None, "plain_string": False}
    )
    assert len(code) == len(raw), (code, raw)
    assert len(no_strings) == len(raw), (no_strings, raw)
    assert code.index("unsafe") == raw.index("unsafe"), (code, raw)
    assert no_strings.index("unsafe") == raw.index("unsafe"), (no_strings, raw)


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
                "    /// # Errors\n"
                "    /// Never fails.\n"
                "    pub fn inner() -> Result<(), ()> {\n"
                "        unsafe {}\n"
                "        Ok(())\n"
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
    # `inner` is documented with `# Errors`, but it is inside the test body: an
    # unfiltered numerator would report 1 here.
    assert counts["with_errors"] == 0, counts


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
    """A checkout under a directory named `target` or `tests` still counts its files.

    Both path exclusions are repo-relative: an absolute-path comparison would drop
    every file here, and a `target/` filter that looked at absolute parts would
    count the generated file below.
    """
    mod = load_module()
    # A checkout whose own path contains `target` and `tests`: an absolute-path
    # comparison would drop every file, so `unsafe == 1` pins that the comparison
    # is repo-relative.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "target" / "tests" / "checkout"
        src = root / "strided-x" / "src"
        src.mkdir(parents=True)
        # 1001 logical lines: an absolute-path filter would drop this file and
        # report 0, and a constant-zero `files_over_1000` would too.
        (src / "real.rs").write_text(
            "pub fn f() {\n    unsafe {}\n}\n" + "// filler\n" * 998, encoding="utf-8"
        )
        counts = mod.count(root)
    assert counts["unsafe"] == 1, counts
    assert counts["files_over_1000"] == 1, counts

    # Exactly 1000 logical lines is not "over 1000": a `>= 1000` threshold would
    # count this file.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "checkout"
        src = root / "strided-x" / "src"
        src.mkdir(parents=True)
        (src / "boundary.rs").write_text(
            "pub fn f() {\n    unsafe {}\n}\n" + "// filler\n" * 997, encoding="utf-8"
        )
        counts = mod.count(root)
    assert counts["unsafe"] == 1, counts
    assert counts["files_over_1000"] == 0, counts

    # A generated file below `strided-x/target/` is excluded from the all-source
    # rows; this root has no `target` component of its own.
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp) / "checkout"
        src = root / "strided-x" / "src"
        src.mkdir(parents=True)
        (src / "real.rs").write_text("pub fn f() {\n    unsafe {}\n}\n", encoding="utf-8")
        generated = root / "strided-x" / "target" / "generated.rs"
        generated.parent.mkdir(parents=True)
        generated.write_text("// filler\n" * 1001, encoding="utf-8")
        counts = mod.count(root)
    assert counts["unsafe"] == 1, counts
    assert counts["files_over_1000"] == 0, counts


def test_doc_block_survives_attributes() -> None:
    """`# Errors` above `#[inline]` / `#[must_use]` still documents the item."""
    counts = count_sources(
        {
            "fixture.rs": (
                "/// # Errors\n"
                "///\n"
                "/// Never returns an error.\n"
                "#[inline]\n"
                "#[must_use]\n"
                "pub fn documented() -> Result<(), ()> {\n"
                "    Ok(())\n"
                "}\n"
                "/// A summary with no error section.\n"
                "pub fn undocumented() -> Result<(), ()> {\n    Ok(())\n}\n"
            ),
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
            "tests.rs": "#[test]\nfn test_only() {\n    unsafe {}\n    unsafe {}\n}\n",
        }
    )
    # One unsafe block in `lib.rs`, two in the test file: keeping `tests.rs`
    # instead of dropping it reports 3, not 1.
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
    # A `///` closing line: taking it as a doc line loses the string's closing
    # quote and swallows the production function after it.
    counts = count_sources(
        {
            "fixture.rs": (
                'const NOTE: &str = "\n'
                '/// ";\n'
                "pub fn live() {\n    unsafe {}\n}\n"
            )
        }
    )
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
            "helpers.rs": (
                "/// # Errors\n"
                "/// Never fails.\n"
                "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n"
            ),
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts
    # The excluded helper is documented with `# Errors`: a numerator taken from
    # unfiltered files would report 1 here.
    assert counts["with_errors"] == 0, counts


def test_a_test_only_module_directory_is_not_production() -> None:
    """The target may be `NAME/mod.rs`, and the subtree below it is test-only too."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod helpers;\n",
            # Counted code in the module file itself, so only the directory
            # exclusion can remove it.
            "helpers/mod.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
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
            "foo/helpers.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
            "helpers.rs": (
                "pub fn live() -> Result<(), ()> {\n"
                "    unsafe {}\n"
                "    unsafe {}\n"
                "    Ok(())\n"
                "}\n"
            ),
        }
    )
    # Two unsafe blocks in the production file, one in the excluded child: an
    # implementation that drops the wrong file reports 1, not 2.
    assert counts["unsafe"] == 2, counts
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
    # A declaration inside an already-excluded file still contributes its target,
    # and an explicit path may point outside the excluded subtree.
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod helpers;\n",
            "helpers.rs": '#[cfg(test)]\n#[path = "outside.rs"]\nmod nested;\n',
            "outside.rs": "pub fn outside() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
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


def test_a_path_attribute_moves_the_test_only_file() -> None:
    """`#[path = "..."] mod NAME;` names the file directly, not `NAME.rs`."""
    counts = count_sources(
        {
            "lib.rs": '#[cfg(test)]\n#[path = "test-support.rs"]\nmod helpers;\n',
            "test-support.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts
    counts = count_sources(
        {
            "lib.rs": 'mod live;\n',
            "live.rs": '#[path = "helpers.rs"]\nmod inner;\n',
            "helpers.rs": "pub fn live() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 1, counts
    assert counts["pub_result"] == 1, counts
    # A `..` component is resolved lexically, because the candidates come from a
    # glob and never carry one.
    counts = count_sources(
        {
            "lib.rs": "mod foo;\n",
            "foo.rs": '#[cfg(test)]\n#[path = "sub/../helper.rs"]\nmod checks;\n',
            "helper.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts
    # A `#[path]` in `foo.rs` resolves against `src/`, not `src/foo/`.
    counts = count_sources(
        {
            "lib.rs": "mod foo;\n",
            "foo.rs": '#[cfg(test)]\n#[path = "helper.rs"]\nmod checks;\n',
            "helper.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_an_inline_test_module_owns_its_directory() -> None:
    """`#[cfg(test)] mod checks { mod helpers; }` owns `checks/helpers.rs`."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod checks {\n    mod helpers;\n}\n",
            "checks/helpers.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_multiline_attribute_does_not_hide_a_test_only_declaration() -> None:
    """An attribute spanning lines still sits between `#[cfg(test)]` and the `mod`."""
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\n#[allow(\n    dead_code\n)]\nmod helpers;\n",
            "helpers.rs": "pub fn helper() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_function_name_ending_in_where_is_not_a_where_clause() -> None:
    """`somewhere` contains `where`; only a standalone keyword ends the return search."""
    counts = count_sources(
        {
            "fixture.rs": (
                "/// # Errors\n"
                "/// Never fails.\n"
                "pub fn somewhere() -> Result<(), ()> {\n    Ok(())\n}\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["with_errors"] == 1, counts


def test_a_char_literal_in_a_signature_does_not_move_bracket_depth() -> None:
    """`b'>'` in an array length is data, so the callback's arrow stays nested."""
    counts = count_sources(
        {
            "fixture.rs": (
                "pub fn accepts(\n"
                "    _: [u8; b'>' as usize],\n"
                "    _: fn() -> Result<(), ()>,\n"
                ") {}\n"
            )
        }
    )
    assert counts["pub_result"] == 0, counts


def test_a_c_raw_string_is_not_code() -> None:
    """`cr##"..."##` is a C string literal, blanked like any other raw string."""
    counts = count_sources(
        {
            "fixture.rs": (
                'const S: &core::ffi::CStr = cr##"\n'
                '" unsafe {} // SAFETY: x // INVARIANT: y\n'
                '"##;\n'
            )
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_docs_of_a_removed_test_module_do_not_credit_the_next_function() -> None:
    """A test module's `# Errors` docs are removed with it, not credited to the next `fn`."""
    counts = count_sources(
        {
            "lib.rs": (
                "/// # Errors\n"
                "/// These docs belong to the test module.\n"
                "#[cfg(test)]\n"
                "mod checks {}\n"
                "pub fn undocumented() -> Result<(), ()> { Ok(()) }\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["with_errors"] == 0, counts


def test_a_test_only_explicit_path_keeps_its_ordinary_children_test_only() -> None:
    """Rust resolves `mod child;` beside the explicitly located `support.rs`."""
    counts = count_sources(
        {
            "lib.rs": '#[cfg(test)]\n#[path = "support.rs"]\nmod checks;\n',
            "support.rs": "mod child;\n",
            "child.rs": "pub fn f() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts
    # The same through a `tests.rs` that the name filter drops.
    counts = count_sources(
        {
            "lib.rs": "#[cfg(test)]\nmod tests;\n",
            "tests.rs": '#[path = "child.rs"]\nmod child;\n',
            "child.rs": "pub fn f() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_declaration_inside_an_inline_module_does_not_exclude_a_production_file() -> None:
    """`mod helpers;` inside `mod outer { .. }` names `outer/helpers.rs`, not `helpers.rs`."""
    counts = count_sources(
        {
            "lib.rs": "mod helpers;\nmod outer {\n    #[cfg(test)]\n    mod helpers;\n}\n",
            "helpers.rs": "pub fn live() -> Result<(), ()> {\n    unsafe {}\n    unsafe {}\n    Ok(())\n}\n",
            "outer/helpers.rs": "pub fn test_only() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 2, counts
    assert counts["pub_result"] == 1, counts


def test_a_path_attribute_before_cfg_test_still_excludes_the_file() -> None:
    """Attribute order does not change the module; the whole attribute list is inspected."""
    counts = count_sources(
        {
            "lib.rs": '#[path = "support.rs"]\n#[cfg(test)]\nmod checks;\n',
            "support.rs": "pub fn f() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_a_trailing_doc_comment_is_not_an_ordinary_comment() -> None:
    """`///` after an item on the same line is documentation, so its markers do not count."""
    counts = count_sources(
        {"lib.rs": "pub struct S; /// // SAFETY: example // INVARIANT: example\npub fn f() {}\n"}
    )
    assert counts["safety"] == 0, counts
    assert counts["invariant"] == 0, counts


def test_an_inline_module_path_moves_its_child() -> None:
    """`#[path]` on an inline module moves the directory its children resolve in.

    rustc resolves `#[path = "elsewhere"] mod checks { mod child; }` to
    `src/elsewhere/child.rs`; naming the child after the module would leave the
    test-only file in the production count.
    """
    counts = count_sources(
        {
            "lib.rs": '#[cfg(test)]\n#[path = "elsewhere"]\nmod checks {\n    mod child;\n}\n',
            "elsewhere/child.rs": "pub fn f() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 0, counts
    assert counts["pub_result"] == 0, counts


def test_an_explicit_path_inside_an_inline_module_keeps_the_file_module_dir() -> None:
    """An inline module in `foo.rs` resolves its explicit path under `foo/`.

    rustc loads `mod inner { #[path = "helper.rs"] mod checks; }` in `src/foo.rs`
    from `src/foo/inner/helper.rs`, not from `src/inner/helper.rs`.
    """
    counts = count_sources(
        {
            "lib.rs": "mod foo;\nmod live;\n",
            "foo.rs": '#[cfg(test)]\nmod inner {\n    #[path = "helper.rs"]\n    mod checks;\n}\n',
            "foo/inner/helper.rs": "pub fn t() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
            "live.rs": (
                "pub fn a() -> Result<(), ()> {\n    unsafe {}\n    unsafe {}\n    Ok(())\n}\n"
            ),
        }
    )
    assert counts["unsafe"] == 2, counts
    assert counts["pub_result"] == 1, counts


def test_docs_of_a_removed_test_module_do_not_credit_a_function_after_a_blank_line() -> None:
    """A blank line between the docs and `#[cfg(test)]` does not separate them."""
    counts = count_sources(
        {
            "lib.rs": (
                "/// # Errors\n"
                "/// These docs belong to the test module.\n"
                "\n"
                "#[cfg(test)]\n"
                "mod checks {}\n"
                "pub fn undocumented() -> Result<(), ()> { Ok(()) }\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["with_errors"] == 0, counts


def test_docs_of_a_removed_test_module_do_not_credit_a_function_after_a_multiline_attribute() -> (
    None
):
    """A multi-line attribute still belongs to the test module it precedes."""
    counts = count_sources(
        {
            "lib.rs": (
                "/// # Errors\n"
                "/// These docs belong to the test module.\n"
                "#[allow(\n    dead_code\n)]\n"
                "#[cfg(test)]\n"
                "mod checks {}\n"
                "pub fn undocumented() -> Result<(), ()> { Ok(()) }\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["with_errors"] == 0, counts


def test_a_test_only_alias_does_not_drop_a_production_file() -> None:
    """A file loaded with and without `#[cfg(test)]` is production."""
    counts = count_sources(
        {
            "lib.rs": (
                '#[path = "shared.rs"]\nmod live;\n'
                '#[cfg(test)]\n#[path = "shared.rs"]\nmod checks;\n'
            ),
            "shared.rs": "pub fn f() -> Result<(), ()> {\n    unsafe {}\n    Ok(())\n}\n",
        }
    )
    assert counts["unsafe"] == 1, counts
    assert counts["pub_result"] == 1, counts


def test_a_heading_that_merely_starts_with_errors_is_not_the_section() -> None:
    """`# ErrorsAreNotDocumented` is a different heading, not `# Errors`."""
    counts = count_sources(
        {
            "lib.rs": (
                "/// # ErrorsAreNotDocumented\n"
                "pub fn f() -> Result<(), ()> { Ok(()) }\n"
            )
        }
    )
    assert counts["pub_result"] == 1, counts
    assert counts["with_errors"] == 0, counts


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
