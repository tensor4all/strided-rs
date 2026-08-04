#!/usr/bin/env python3
"""Self-contained tests for scripts/repository-rules-review.py.

Run with `python3 scripts/test-repository-rules-review.py`; no pytest needed.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "repository-rules-review.py"


def load_module():
    spec = importlib.util.spec_from_file_location("repository_rules_review", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_diff(path: str, added: list[str], *, start: int = 1) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            "index abc..def 100644",
            f"--- a/{path}",
            f"+++ b/{path}",
            f"@@ -{start},1 +{start},{len(added) + 1} @@",
            " unchanged",
            *(f"+{line}" for line in added),
        ]
    )


# --- diff parsing ------------------------------------------------------------


def test_added_line_numbers_tracks_context_offsets() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/foo.rs b/foo.rs",
            "index abc..def 100644",
            "--- a/foo.rs",
            "+++ b/foo.rs",
            "@@ -1,3 +1,4 @@",
            " unchanged",
            "+added",
            " context",
        ]
    )
    lines = mod.added_line_numbers(mod.added_lines_with_text(diff))
    assert lines["foo.rs"] == {2}


def test_added_lines_with_text_matches_line_numbers() -> None:
    mod = load_module()
    diff = make_diff("strided-view/src/view.rs", ["let a = 1;", "let b = 2;"])
    entries = mod.added_lines_with_text(diff)
    assert entries["strided-view/src/view.rs"] == [(2, "let a = 1;"), (3, "let b = 2;")]
    numbers = mod.added_line_numbers(entries)
    assert numbers["strided-view/src/view.rs"] == {2, 3}


def test_added_lines_with_text_ignores_deleted_file() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/gone.rs b/gone.rs",
            "--- a/gone.rs",
            "+++ /dev/null",
            "@@ -1,1 +0,0 @@",
            "-removed",
        ]
    )
    assert mod.added_lines_with_text(diff) == {}


# --- model defaults ----------------------------------------------------------


def test_default_deepseek_model_uses_current_v4_name() -> None:
    mod = load_module()
    assert mod.DEFAULT_MODEL == "deepseek-v4-pro"
    assert mod.DEFAULT_API_URL == "https://api.deepseek.com/chat/completions"


# --- finding filtering -------------------------------------------------------


def test_filter_findings_drops_unchanged_files() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="other.rs",
        line=1,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {1}}) == []


def test_filter_findings_keeps_added_line() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=2,
        summary="test",
        detail="detail",
    )
    assert len(mod.filter_findings([finding], ["foo.rs"], {"foo.rs": {2}})) == 1


def test_filter_findings_drops_line_finding_without_added_lines() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="deleted.rs",
        line=4,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([finding], ["deleted.rs"], {}) == []


def test_filter_findings_drops_file_level_block_finding() -> None:
    mod = load_module()
    block = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    warn = mod.Finding(
        id="w",
        severity="warn",
        rule_section="Public Surface Discipline",
        file="foo.rs",
        line=None,
        summary="test",
        detail="detail",
    )
    assert mod.filter_findings([block, warn], ["foo.rs"], {"foo.rs": {1}}) == [warn]


def test_filter_findings_drops_global_llm_finding_when_disallowed() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Surface Discipline",
        file="",
        line=None,
        summary="test",
        detail="detail",
    )
    kept = mod.filter_findings(
        [finding], ["foo.rs"], {"foo.rs": {1}}, allow_global=False
    )
    assert kept == []


def test_reconcile_verdict_only_blocks_fail() -> None:
    mod = load_module()
    warn = mod.Finding("w", "warn", "s", "f", 1, "s", "d")
    block = mod.Finding("b", "block", "s", "f", 1, "s", "d")
    assert mod.reconcile_verdict([warn]) == "pass"
    assert mod.reconcile_verdict([warn, block]) == "fail"


def test_merge_findings_prefers_block_over_warn() -> None:
    mod = load_module()
    warn = mod.Finding("dup", "warn", "s", "f.rs", 3, "same", "d")
    block = mod.Finding("dup", "block", "s", "f.rs", 3, "same", "d")
    merged = mod.merge_findings([warn, block])
    assert len(merged) == 1
    assert merged[0].severity == "block"


# --- rule section routing ----------------------------------------------------


def test_select_rule_sections_always_includes_public_surface_and_boundary() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["Cargo.toml"])
    assert "Public Surface Discipline" in sections
    assert "Public Boundary Safety" in sections


def test_select_rule_sections_routes_kernel_paths_to_threading() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["strided-kernel/src/threading.rs"])
    assert "CPU Threading Contract" in sections
    assert "Unsafe And Fast-Path Boundaries" in sections


def test_select_rule_sections_routes_perm_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["strided-perm/src/hptt/execute.rs"])
    assert "CPU Threading Contract" in sections
    assert "Layout And Copy Semantics" in sections


def test_select_rule_sections_routes_view_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["strided-view/src/view.rs"])
    assert "Layout And Copy Semantics" in sections
    assert "Materialization And Copies" in sections


def test_select_rule_sections_routes_bench_paths() -> None:
    mod = load_module()
    sections = mod.select_rule_sections(["strided-kernel/benches/map.rs"])
    assert "Performance And Benchmark Discipline" in sections


def test_select_rule_sections_routes_retired_crates_to_freeze() -> None:
    mod = load_module()
    for crate in mod.RETIRED_CRATES:
        sections = mod.select_rule_sections([f"{crate}/src/lib.rs"])
        assert "Retired Crate Freeze" in sections, crate
    assert "Retired Crate Freeze" in mod.select_rule_sections(
        ["deprecated/benches/strided_bench.rs"]
    )


def test_select_rule_sections_excludes_human_only_sections() -> None:
    mod = load_module()
    sections = set(mod.select_rule_sections(["strided-kernel/src/threading.rs"]))
    assert sections.isdisjoint(mod.HUMAN_ONLY_SECTIONS)


def test_every_rule_section_is_reachable() -> None:
    """A new REPOSITORY_RULES section must be routed, always-on, or human-only.

    Without this, adding a section silently makes it invisible to the reviewer.
    """
    mod = load_module()
    documented = set(mod.parse_repository_rules_sections())
    routed = set(mod.ALWAYS_SECTIONS) | set(mod.HUMAN_ONLY_SECTIONS)
    for _pattern, names in mod.SECTION_TRIGGERS:
        routed |= set(names)
    assert documented <= routed, f"unrouted rule sections: {sorted(documented - routed)}"
    assert routed <= documented, f"routing names a missing section: {sorted(routed - documented)}"


def test_build_rules_payload_returns_requested_section_bodies() -> None:
    mod = load_module()
    payload = mod.build_rules_payload(["CPU Threading Contract"])
    assert payload.startswith("## CPU Threading Contract")
    assert "Public Surface Discipline" not in payload


# --- retired-crate freeze ----------------------------------------------------


def test_retired_freeze_blocks_source_change() -> None:
    mod = load_module()
    diff = make_diff("strided-einsum2/src/util.rs", ["fn faster() -> usize { 1 }"])
    findings = mod.deterministic_checks(
        ["strided-einsum2/src/util.rs"], added=mod.added_lines_with_text(diff)
    )
    assert len(findings) == 1
    assert findings[0].severity == "block"
    assert findings[0].id == "retired-crate-freeze"
    assert "strided-einsum2/src/util.rs:2" in findings[0].detail


def test_retired_freeze_allows_deprecation_notices() -> None:
    mod = load_module()
    diff = make_diff(
        "strided-opteinsum/src/lib.rs",
        [
            "//! Deprecated: use tenferro-einsum instead.",
            "/// Migration pointer.",
            '#[deprecated(note = "see strided-rs#199")]',
            "",
        ],
    )
    findings = mod.deterministic_checks(
        ["strided-opteinsum/src/lib.rs"], added=mod.added_lines_with_text(diff)
    )
    assert findings == []


def test_retired_freeze_exempts_block_comments_but_not_dereference() -> None:
    mod = load_module()
    diff = make_diff(
        "strided-einsum2/src/uninit.rs",
        [
            "/** Deprecated. */",
            " * continuation",
            " */",
            "*dst = value;",
        ],
    )
    findings = mod.deterministic_checks(
        ["strided-einsum2/src/uninit.rs"], added=mod.added_lines_with_text(diff)
    )
    assert len(findings) == 1
    detail = findings[0].detail
    assert "*dst = value;" in detail
    assert "continuation" not in detail


def test_retired_freeze_allows_readme_and_manifest() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            make_diff("mdarray-opteinsum/README.md", ["> **Deprecated.**"]),
            make_diff("mdarray-opteinsum/Cargo.toml", ['description = "deprecated"']),
        ]
    )
    findings = mod.deterministic_checks(
        ["mdarray-opteinsum/README.md", "mdarray-opteinsum/Cargo.toml"],
        added=mod.added_lines_with_text(diff),
    )
    assert findings == []


def test_retired_freeze_ignores_retained_crates() -> None:
    mod = load_module()
    diff = make_diff("strided-kernel/src/threading.rs", ["fn faster() -> usize { 1 }"])
    findings = mod.deterministic_checks(
        ["strided-kernel/src/threading.rs"], added=mod.added_lines_with_text(diff)
    )
    assert findings == []


def test_retired_path_prefixes_cover_all_retired_crates() -> None:
    mod = load_module()
    for crate in mod.RETIRED_CRATES:
        assert mod.is_retired_path(f"{crate}/src/lib.rs"), crate
    assert mod.is_retired_path("deprecated/benches/strided_bench.rs")
    assert not mod.is_retired_path("strided-view/src/view.rs")
    # Prefix matching must not catch a retained crate whose name shares a stem.
    assert not mod.is_retired_path("strided-einsum2-notes.md")


# --- model response handling -------------------------------------------------


def test_extract_json_payload_strips_fence() -> None:
    mod = load_module()
    payload = mod.extract_json_payload('```json\n{"verdict": "pass"}\n```')
    assert payload == {"verdict": "pass"}


def test_extract_json_payload_reports_malformed_embedded_object() -> None:
    mod = load_module()
    try:
        mod.extract_json_payload("noise {not json} tail")
    except ValueError as err:
        assert "not valid JSON" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_parse_findings_caps_model_output() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {
                "id": f"f{index}",
                "severity": "block",
                "rule_section": "Public Boundary Safety",
                "file": "strided-view/src/view.rs",
                "line": index + 1,
                "summary": "s",
                "detail": "d",
            }
            for index in range(mod.MAX_FINDINGS_PER_CHUNK + 5)
        ],
    }
    verdict, findings = mod.parse_findings(raw)
    assert verdict == "fail"
    assert len(findings) == mod.MAX_FINDINGS_PER_CHUNK


def test_parse_findings_normalizes_common_severity_aliases() -> None:
    mod = load_module()
    raw = {
        "verdict": "fail",
        "findings": [
            {"id": "a", "severity": "CRITICAL", "file": "f.rs", "line": 1},
            {"id": "b", "severity": "info", "file": "f.rs", "line": 2},
            {"id": "c", "severity": "nonsense", "file": "f.rs", "line": 3},
        ],
    }
    _, findings = mod.parse_findings(raw)
    assert [item.severity for item in findings] == ["block", "warn", "warn"]


def test_parse_findings_rejects_non_integer_line() -> None:
    mod = load_module()
    raw = {"verdict": "pass", "findings": [{"id": "a", "line": "3"}]}
    try:
        mod.parse_findings(raw)
    except ValueError as err:
        assert "line must be an integer" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_parse_findings_rejects_unknown_verdict() -> None:
    mod = load_module()
    try:
        mod.parse_findings({"verdict": "maybe", "findings": []})
    except ValueError as err:
        assert "verdict" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_llm_response_error_finding_blocks_with_diagnostic() -> None:
    mod = load_module()
    finding = mod.llm_response_error_finding(ValueError("bad json"))
    assert finding.severity == "block"
    assert "ValueError: bad json" in finding.detail


# --- diff chunking -----------------------------------------------------------


def test_split_diff_chunks_respects_limit() -> None:
    mod = load_module()
    # Each file stays under the per-file limit, so only the aggregate limit splits.
    piece = "x" * (mod.MAX_FILE_DIFF_CHARS - 10)
    per_chunk = mod.MAX_DIFF_CHARS // len(piece)
    files = {f"f{index}.rs": piece for index in range(per_chunk + 1)}
    chunks = mod.split_diff_chunks(files)
    assert len(chunks) == 2
    assert all(len(chunk) <= mod.MAX_DIFF_CHARS for chunk in chunks)


def test_split_large_file_diff_preserves_file_header() -> None:
    mod = load_module()
    header = [
        "diff --git a/big.rs b/big.rs",
        "--- a/big.rs",
        "+++ b/big.rs",
    ]
    body = "\n".join(
        f"@@ -{index},1 +{index},1 @@\n+{'y' * 1000}"
        for index in range(1, 120)
    )
    chunks = mod.split_large_file_diff("\n".join([*header, body]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("diff --git a/big.rs b/big.rs")
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS


def test_split_large_file_diff_splits_single_overlong_line() -> None:
    mod = load_module()
    header = [
        "diff --git a/big.rs b/big.rs",
        "--- a/big.rs",
        "+++ b/big.rs",
    ]
    line = "+" + "z" * (mod.MAX_FILE_DIFF_CHARS * 2)
    chunks = mod.split_large_file_diff("\n".join([*header, "@@ -1,1 +1,1 @@", line]))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.startswith("diff --git a/big.rs b/big.rs")
        assert len(chunk) <= mod.MAX_FILE_DIFF_CHARS


def test_call_deepseek_retries_transient_network_errors() -> None:
    """socket.timeout is only a TimeoutError alias from Python 3.10 on."""
    import socket
    import urllib.request

    mod = load_module()
    calls = {"n": 0}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"verdict\\":\\"pass\\"}"}}]}'

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise socket.timeout("The read operation timed out")
        return FakeResponse()

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = fake_urlopen
    mod.time.sleep = lambda _seconds: None
    try:
        payload = mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep

    assert calls["n"] == 2
    assert payload == {"verdict": "pass"}


def test_call_deepseek_reraises_after_retries_exhausted() -> None:
    import socket
    import urllib.request

    mod = load_module()

    def always_timeout(request, timeout=None):
        raise socket.timeout("nope")

    original_urlopen = urllib.request.urlopen
    original_sleep = mod.time.sleep
    urllib.request.urlopen = always_timeout
    mod.time.sleep = lambda _seconds: None
    try:
        mod.call_deepseek(
            api_key="k",
            model="m",
            api_url="https://example.invalid",
            system_prompt="s",
            user_content="u",
            timeout=1.0,
        )
    except OSError:
        pass
    else:
        raise AssertionError("expected the timeout to propagate")
    finally:
        urllib.request.urlopen = original_urlopen
        mod.time.sleep = original_sleep


# --- secret handling ---------------------------------------------------------


def test_redact_sensitive_text_masks_common_secret_forms() -> None:
    mod = load_module()
    text = "\n".join(
        [
            "ghp_abcdefghijklmnopqrstuvwxyz0123",
            "AKIAABCDEFGHIJKLMNOP",
            "api_key = supersecretvalue",
            "Authorization: Bearer abcdefghijklmnopqrst",
        ]
    )
    redacted = mod.redact_sensitive_text(text)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "supersecretvalue" not in redacted
    assert redacted.count("[REDACTED_SECRET]") >= 4


def test_contains_sensitive_text_ignores_env_lookup_code() -> None:
    mod = load_module()
    assert not mod.contains_sensitive_text(
        'let key = std::env::var("DEEPSEEK_API_KEY")?;'
    )
    assert not mod.contains_sensitive_text("DEEPSEEK_API_KEY: ${{ secrets.KEY }}")


def test_contains_sensitive_text_flags_quoted_credential() -> None:
    mod = load_module()
    assert mod.contains_sensitive_text('let api_key = "abcdefghijklmnop";')


def test_sensitive_diff_finding_checks_added_lines_only() -> None:
    mod = load_module()
    diff = "\n".join(
        [
            "diff --git a/a.rs b/a.rs",
            "--- a/a.rs",
            "+++ b/a.rs",
            "@@ -1,2 +1,2 @@",
            " let token = ghp_abcdefghijklmnopqrstuvwxyz0123;",
            "+let clean = 1;",
        ]
    )
    assert mod.sensitive_diff_finding(diff) is None


def test_sensitive_diff_finding_reports_added_match_location() -> None:
    mod = load_module()
    diff = make_diff(
        "a.rs", ["let clean = 1;", "let t = ghp_abcdefghijklmnopqrstuvwxyz0123;"]
    )
    finding = mod.sensitive_diff_finding(diff)
    assert finding is not None
    assert finding.severity == "block"
    assert (finding.file, finding.line) == ("a.rs", 3)


# --- reporting ---------------------------------------------------------------


def test_summarize_llm_review_computes_dropped_count() -> None:
    mod = load_module()
    summary = mod.summarize_llm_review(
        chunk_sizes=[100, 200], elapsed_seconds=1.25, returned_count=5, kept_count=2
    )
    assert "2 chunk(s)" in summary
    assert "3 dropped" in summary


def test_format_report_includes_llm_summary_line() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base",
        head="head",
        verdict="pass",
        findings=[],
        waived=False,
        llm_summary="LLM review: 1 chunk(s)",
    )
    assert "LLM review: 1 chunk(s)" in report
    assert "No findings." in report


def test_format_report_omits_llm_summary_when_absent() -> None:
    mod = load_module()
    report = mod.format_report(
        base="base", head="head", verdict="pass", findings=[], waived=False
    )
    assert "LLM review:" not in report


def test_format_report_lists_findings_with_location() -> None:
    mod = load_module()
    finding = mod.Finding(
        id="x",
        severity="block",
        rule_section="Public Boundary Safety",
        file="strided-view/src/view.rs",
        line=42,
        summary="missing validation",
        detail="detail line",
    )
    report = mod.format_report(
        base="base", head="head", verdict="fail", findings=[finding], waived=False
    )
    assert "[block] x (Public Boundary Safety) strided-view/src/view.rs:42" in report
    assert "detail line" in report


# --- prompt and rules files --------------------------------------------------


def test_prompt_file_exists_and_requires_json_only() -> None:
    mod = load_module()
    assert mod.PROMPT_PATH.is_file()
    text = mod.PROMPT_PATH.read_text(encoding="utf-8")
    assert "JSON only" in text
    assert "untrusted data" in text


def test_rules_file_documents_the_retirement() -> None:
    mod = load_module()
    freeze = mod.parse_repository_rules_sections()["Retired Crate Freeze"]
    for crate in mod.RETIRED_CRATES:
        assert crate in freeze, crate
    assert "199" in freeze


def main() -> int:
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"repository-rules-review: {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
