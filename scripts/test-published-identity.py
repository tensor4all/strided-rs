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

    def run(self, ref: str, published=None) -> tuple[int, str]:
        out = io.StringIO()
        with mock.patch.object(pi, "published", published or (lambda n, v: None)), \
                mock.patch("sys.argv", ["published_identity.py", str(self.root), "--ref", ref]), \
                contextlib.redirect_stdout(out):
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

    def test_failed_read_of_a_shipped_file_is_not_green(self) -> None:
        # GPT review on #411 (round 1): a new shipped file whose read fails must
        # not drop out of the comparison and leave the run green.
        blob = self.fx.package()
        self.fx.write("src/new.rs", "pub fn extra() {}\n")
        self.fx.commit("new shipped file, same version")
        real = pathlib.Path.read_bytes

        def flaky(path):
            if str(path).endswith("src/new.rs"):
                raise OSError("unable to read")
            return real(path)

        with mock.patch.object(pi.pathlib.Path, "read_bytes", flaky):
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
