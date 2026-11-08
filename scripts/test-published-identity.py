#!/usr/bin/env python3
"""Offline tests for published_identity.py.

Each test builds a one-crate git repository in a temporary directory, packages
it with `cargo package --no-verify --offline` to stand in for the crates.io
artefact, and replaces `published()` with a function that returns that archive.
Nothing touches the network. Requires `git` and `cargo`.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("published_identity", HERE / "published_identity.py")
pi = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(pi)

MANIFEST = """[workspace]
members = ["demo"]
resolver = "2"
"""
CRATE_MANIFEST = """[package]
name = "demo"
version = "0.1.0"
edition = "2021"
license = "MIT"
description = "fixture"
"""


def sh(cwd: pathlib.Path, *cmd: str) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


class Fixture:
    def __init__(self, root: pathlib.Path) -> None:
        self.root = root
        sh(root, "git", "init", "-q", "-b", "main")
        sh(root, "git", "config", "user.email", "t@example.invalid")
        sh(root, "git", "config", "user.name", "t")
        (root / "Cargo.toml").write_text(MANIFEST)
        (root / "demo" / "src").mkdir(parents=True)
        (root / "demo" / "Cargo.toml").write_text(CRATE_MANIFEST)
        self.write("src/lib.rs", "pub fn value() -> u32 { 1 }\n")
        self.commit("published state")

    def write(self, rel: str, text: str) -> None:
        (self.root / "demo" / rel).write_text(text)

    def commit(self, msg: str) -> str:
        sh(self.root, "git", "add", "-A")
        sh(self.root, "git", "commit", "-q", "-m", msg)
        return sh(self.root, "git", "rev-parse", "HEAD").strip()

    def package(self) -> bytes:
        sh(self.root, "cargo", "package", "-q", "-p", "demo", "--no-verify", "--offline")
        return (self.root / "target" / "package" / "demo-0.1.0.crate").read_bytes()

    def run(self, ref: str, published=None, real_network: bool = False) -> tuple[int, str]:
        """Run main(). `published()` is replaced unless `real_network` is set,
        in which case the caller mocks the transport underneath it."""
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            if not real_network:
                stack.enter_context(mock.patch.object(pi, "published", published or (lambda n, v: None)))
            stack.enter_context(mock.patch("sys.argv", ["published_identity.py", str(self.root), "--ref", ref]))
            stack.enter_context(contextlib.redirect_stdout(out))
            rc = pi.main()
        return rc, out.getvalue()


class PublishedIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(pathlib.Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_identical_tree_is_ok(self) -> None:
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_unpublished_version_is_green(self) -> None:
        rc, out = self.fx.run("HEAD")
        self.assertEqual(rc, 0, out)
        self.assertIn("unpublished demo 0.1.0", out)

    def test_changed_file_under_same_version_is_red(self) -> None:
        blob = self.fx.package()
        self.fx.write("src/lib.rs", "pub fn value() -> u32 { 2 }\n")
        self.fx.commit("same version, new content")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("differs: src/lib.rs", out)

    def test_file_deleted_since_publication_is_red(self) -> None:
        # GPT review on #411: a build script that sets a cfg; deleting it
        # under the same version flips `value()` from true to false.
        self.fx.write("build.rs", 'fn main() { println!("cargo:rustc-cfg=build_flag"); }\n')
        self.fx.write("src/lib.rs", "pub fn value() -> bool { cfg!(build_flag) }\n")
        self.fx.commit("add build.rs")
        blob = self.fx.package()
        (self.fx.root / "demo" / "build.rs").unlink()
        self.fx.commit("delete build.rs, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("only in published .crate: build.rs", out)

    def test_relative_ref_names_the_same_commit(self) -> None:
        blob = self.fx.package()
        target = sh(self.fx.root, "git", "rev-parse", "HEAD").strip()
        self.fx.write("src/lib.rs", "pub fn value() -> u32 { 2 }\n")
        self.fx.commit("later commit")
        rc_sha, out_sha = self.fx.run(target, lambda n, v: blob)
        rc_rel, out_rel = self.fx.run("HEAD~1", lambda n, v: blob)
        self.assertEqual((rc_sha, rc_rel), (0, 0), out_sha + out_rel)

    def test_failed_packaging_of_the_tree_is_not_green(self) -> None:
        # GPT review on #411 (round 1, git-read case): if the tree side cannot
        # be produced, the run must not be green. The tree side is now Cargo's
        # own archive, so the failure point is `cargo package`.
        blob = self.fx.package()
        self.fx.write("src/new.rs", "pub fn extra() {}\n")
        self.fx.commit("new shipped file, same version")
        real = subprocess.run

        def broken(cmd, *args, **kwargs):
            if cmd[:2] == ["cargo", "package"]:
                return subprocess.CompletedProcess(cmd, 101, "", "error: failed to read\n")
            return real(cmd, *args, **kwargs)

        with mock.patch.object(pi.subprocess, "run", broken):
            rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)

    def test_unchanged_symlink_is_ok(self) -> None:
        # GPT review on #411 (round 2): Cargo archives the symlink target.
        (self.fx.root / "demo" / "src" / "alias.rs").symlink_to("lib.rs")
        self.fx.commit("add symlink")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_unchanged_external_readme_is_ok(self) -> None:
        # GPT review on #411 (round 2): `readme = "../README.md"` ships as README.md.
        (self.fx.root / "README.md").write_text("# shared readme\n")
        self.fx.write("Cargo.toml", CRATE_MANIFEST + 'readme = "../README.md"\n')
        self.fx.commit("shared readme")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        (self.fx.root / "README.md").write_text("# changed\n")
        self.fx.commit("readme changed, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("differs: README.md", out)

    def test_changed_crate_local_readme_shadowing_an_external_one_is_red(self) -> None:
        # GPT review on #411 (round 4): with both ../README.md and demo/README.md,
        # Cargo archives the crate-local one; changing it must be a MISMATCH.
        (self.fx.root / "README.md").write_text("# same\n")
        self.fx.write("README.md", "# same\n")
        self.fx.write("Cargo.toml", CRATE_MANIFEST + 'readme = "../README.md"\n')
        self.fx.commit("both readmes")
        blob = self.fx.package()
        self.fx.write("README.md", "# changed locally\n")
        self.fx.commit("crate-local readme changed, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("README.md", out)

    def test_unchanged_external_readme_symlink_into_the_crate_is_ok(self) -> None:
        # GPT review on #411 (round 4): ../README.md -> demo/docs/GUIDE.md.
        (self.fx.root / "demo" / "docs").mkdir()
        self.fx.write("docs/GUIDE.md", "# guide\n")
        (self.fx.root / "README.md").symlink_to("demo/docs/GUIDE.md")
        self.fx.write("Cargo.toml", CRATE_MANIFEST + 'readme = "../README.md"\n')
        self.fx.commit("external readme symlink pointing into the crate")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_cargo_warning_on_stderr_does_not_break_the_check(self) -> None:
        # GPT review on #411 (round 4): a legacy `.cargo/config` makes cargo
        # warn on stderr while printing valid JSON on stdout.
        (self.fx.root / ".cargo").mkdir()
        (self.fx.root / ".cargo" / "config").write_text("[net]\noffline = true\n")
        self.fx.commit("legacy cargo config")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def _docs_readme_symlink_fixture(self) -> None:
        # demo/docs/README.md -> ../../GUIDE.md, readme = "docs/README.md",
        # plus an unrelated root README.md and demo/README.md with other bytes.
        (self.fx.root / "GUIDE.md").write_text("# guide\n")
        (self.fx.root / "README.md").write_text("# unrelated root readme\n")
        (self.fx.root / "demo" / "docs").mkdir()
        (self.fx.root / "demo" / "docs" / "README.md").symlink_to("../../GUIDE.md")
        self.fx.write("README.md", "# unrelated crate readme\n")
        self.fx.write("Cargo.toml", CRATE_MANIFEST + 'readme = "docs/README.md"\n')
        self.fx.commit("docs/README.md symlink to a root-level GUIDE.md")

    def test_unchanged_readme_symlink_to_a_differently_named_target_is_ok(self) -> None:
        # GPT review on #411 (18:10Z, case a): no false MISMATCH.
        self._docs_readme_symlink_fixture()
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_changed_unrelated_crate_readme_next_to_a_readme_symlink_is_red(self) -> None:
        # GPT review on #411 (18:10Z, case b): changing demo/README.md under the
        # same version must be a MISMATCH.
        self._docs_readme_symlink_fixture()
        blob = self.fx.package()
        self.fx.write("README.md", "# changed\n")
        self.fx.commit("demo/README.md changed, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("differs: README.md", out)

    def test_unchanged_file_with_a_newline_in_its_name_is_ok(self) -> None:
        # GPT review on #411 (18:10Z, item 2): `notes\nmore` is one file.
        self.fx.write("notes\nmore", "x\n")
        self.fx.commit("file name with a newline")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_crc_damaged_archive_is_not_green(self) -> None:
        # GPT review on #411 (round 5): flip one bit of the gzip CRC32 trailer.
        blob = self.fx.package()
        damaged = blob[:-8] + bytes([blob[-8] ^ 1]) + blob[-7:]
        rc, out = self.fx.run("HEAD", lambda n, v: damaged)
        self.assertEqual(rc, 2, out)

    def test_truncated_archive_is_not_green(self) -> None:
        # GPT review on #411 (round 5): drop the 8-byte gzip trailer.
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob[:-8])
        self.assertEqual(rc, 2, out)

    def test_default_feature_flip_under_same_version_is_red(self) -> None:
        # GPT review on #411 (round 5): a manifest-only change that flips
        # `value()` from 0 to 1 must be a MISMATCH (seen via Cargo.toml.orig).
        self.fx.write("src/lib.rs",
                      'pub fn value() -> u32 { cfg!(feature = "alternate") as u32 }\n')
        manifest = CRATE_MANIFEST + '\n[features]\nalternate = []\ndefault = []\n'
        self.fx.write("Cargo.toml", manifest)
        self.fx.commit("published feature configuration")
        blob = self.fx.package()
        self.fx.write("Cargo.toml", manifest.replace('default = []', 'default = ["alternate"]'))
        self.fx.commit("change behavior without bump")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("differs: Cargo.toml.orig", out)

    def test_workspace_inherited_change_is_out_of_scope(self) -> None:
        # Documented limitation: a change made only in the workspace root's
        # [workspace.package] is invisible in Cargo.toml.orig. This pins the
        # limitation so that a future fix updates the docstring and this test.
        (self.fx.root / "Cargo.toml").write_text(MANIFEST + '\n[workspace.package]\nedition = "2021"\n')
        self.fx.write("Cargo.toml", CRATE_MANIFEST.replace('edition = "2021"', 'edition.workspace = true'))
        self.fx.commit("inherit edition")
        blob = self.fx.package()
        (self.fx.root / "Cargo.toml").write_text(MANIFEST + '\n[workspace.package]\nedition = "2018"\n')
        self.fx.commit("root-only change, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)

    def test_missing_cargo_is_not_a_mismatch(self) -> None:
        real = subprocess.run

        def no_cargo(cmd, *args, **kwargs):
            if cmd[0] == "cargo":
                raise FileNotFoundError(2, "No such file or directory", "cargo")
            return real(cmd, *args, **kwargs)

        with mock.patch.object(pi.subprocess, "run", no_cargo):
            rc, out = self.fx.run("HEAD")
        self.assertEqual(rc, 2, out)

    def test_cargo_metadata_timeout_is_not_a_mismatch(self) -> None:
        real = subprocess.run

        def slow(cmd, *args, **kwargs):
            if cmd[:2] == ["cargo", "metadata"]:
                raise subprocess.TimeoutExpired(cmd, 120)
            return real(cmd, *args, **kwargs)

        with mock.patch.object(pi.subprocess, "run", slow):
            rc, out = self.fx.run("HEAD")
        self.assertEqual(rc, 2, out)

    def test_malformed_archive_is_not_a_mismatch(self) -> None:
        rc, out = self.fx.run("HEAD", lambda n, v: b"not a gzip")
        self.assertEqual(rc, 2, out)

    def test_network_failure_is_not_green(self) -> None:
        def down(name, ver):
            raise pi.CheckError("GET https://crates.io/...: timed out")

        rc, out = self.fx.run("HEAD", down)
        self.assertEqual(rc, 2, out)

    def test_transport_failure_through_the_real_call_chain_is_not_green(self) -> None:
        # GPT review on #411 (round 3): mock urlopen, not published()/fetch(),
        # so that misclassifying a DNS outage as 404 would turn this test red.
        import urllib.error

        def outage(*args, **kwargs):
            raise urllib.error.URLError("DNS outage")

        with mock.patch.object(pi.urllib.request, "urlopen", outage):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 2, out)
        self.assertNotIn("unpublished", out)

    def _fake_crates_io(self, api_status: int, crate_blob: bytes | None = None):
        """A urlopen replacement: the API URL answers `api_status`, the static
        download answers 200 with `crate_blob`. Non-2xx statuses are raised as
        HTTPError, exactly as urllib does, so fetch()'s HTTPError branch runs."""
        import urllib.error

        class Resp(io.BytesIO):
            def __init__(self, body: bytes) -> None:
                super().__init__(body)
                self.status = 200

        def urlopen(req, *args, **kwargs):
            url = req.full_url
            if "/api/v1/crates/" in url:
                if api_status != 200:
                    raise urllib.error.HTTPError(url, api_status, "status", {}, None)
                return Resp(b"{}")
            if "static.crates.io" in url and crate_blob is not None:
                return Resp(crate_blob)
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)

        return mock.patch.object(pi.urllib.request, "urlopen", urlopen)

    def test_api_503_through_urlopen_is_not_green(self) -> None:
        # GPT review on #411 (round 6): an HTTPError 503 from the API must be
        # exit 2. Mutating fetch()'s HTTPError branch to `return 404, b""`
        # turns this red; mocking fetch() itself could not see that.
        with self._fake_crates_io(503):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 2, out)
        self.assertNotIn("unpublished", out)

    def test_api_404_through_urlopen_is_unpublished(self) -> None:
        with self._fake_crates_io(404):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("unpublished demo 0.1.0", out)

    def test_published_archive_with_different_source_through_urlopen_is_red(self) -> None:
        # GPT review on #411 (round 6): API 200, then a valid archive whose
        # source differs from the tree, must be exit 1. A mutation that
        # returns None after the API check (disabling every comparison)
        # turns this red.
        blob = self.fx.package()
        self.fx.write("src/lib.rs", "pub fn value() -> u32 { 2 }\n")
        self.fx.commit("same version, new content")
        with self._fake_crates_io(200, blob):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 1, out)
        self.assertIn("differs: src/lib.rs", out)

    def test_whitespace_in_a_shipped_name_is_kept(self) -> None:
        # GPT review on #411 (round 3): "notes" and "notes " are two files.
        self.fx.write("notes", "a\n")
        self.fx.commit("notes")
        blob = self.fx.package()
        self.fx.write("notes ", "b\n")
        self.fx.commit("second file whose name ends in a space")
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("only in", out)

    def test_unchanged_external_readme_symlink_is_ok(self) -> None:
        # GPT review on #411 (round 3): README.md -> GUIDE.md at the repo root;
        # Cargo archives it as README.md.
        (self.fx.root / "GUIDE.md").write_text("# guide\n")
        (self.fx.root / "README.md").symlink_to("GUIDE.md")
        self.fx.write("Cargo.toml", CRATE_MANIFEST + 'readme = "../README.md"\n')
        self.fx.commit("readme symlink outside the crate")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_only_404_means_unpublished(self) -> None:
        with mock.patch.object(pi, "fetch", lambda url: (404, b"")):
            self.assertIsNone(pi.published("demo", "0.1.0"))
        with mock.patch.object(pi, "fetch", lambda url: (403, b"")):
            with self.assertRaises(pi.CheckError):
                pi.published("demo", "0.1.0")

    def test_unknown_crate_name_is_not_green(self) -> None:
        out = io.StringIO()
        argv = ["published_identity.py", str(self.fx.root), "--crate", "nope"]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            self.assertEqual(pi.main(), 2, out.getvalue())


if __name__ == "__main__":
    unittest.main()
