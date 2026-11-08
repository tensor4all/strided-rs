#!/usr/bin/env python3
"""Offline tests for published_identity.py.

Each test builds a one-crate git repository in a temporary directory, packages
it with `cargo package --no-verify --offline` to stand in for the crates.io
artefact, and replaces `published()` with a function that returns that archive.
Nothing touches the network. Requires `git` and `cargo`.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import importlib.util
import io
import json
import pathlib
import subprocess
import sys
import tarfile
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


def repack(blob: bytes, replace: dict[str, bytes | None]) -> bytes:
    """`blob` as a .crate again, with the named members' bytes replaced (or
    dropped, when the replacement is None)."""
    src = tarfile.open(fileobj=io.BytesIO(gzip.decompress(blob)), mode="r:")
    buf = io.BytesIO()
    with src, tarfile.open(fileobj=buf, mode="w") as dst:
        for m in src.getmembers():
            if not m.isfile():
                continue
            data = replace[m.name] if m.name in replace else src.extractfile(m).read()
            if data is None:
                continue
            info = tarfile.TarInfo(m.name)
            info.size, info.mtime = len(data), m.mtime
            dst.addfile(info, io.BytesIO(data))
    body = buf.getvalue()
    # tarfile pads to a 10240-byte record; Cargo stops after the two
    # end-of-archive zero blocks, and the checker pins that.
    end = -(-len(body.rstrip(b"\0")) // 512) * 512
    return gzip.compress(body[:end + 1024])


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

    def test_damaged_member_header_is_not_green(self) -> None:
        # #415: tarfile stops silently at a header whose checksum is wrong, so a
        # damaged archive reads as a SHORTER member list. Here the last member's
        # header is damaged and that file is gone from the tree, so the short
        # parse "matches" and the checker used to report ok without looking.
        self.fx.write("zz.txt", "zz\n")
        self.fx.commit("add zz.txt")
        raw = bytearray(gzip.decompress(self.fx.package()))
        with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as tf:
            offset = tf.getmember("demo-0.1.0/zz.txt").offset
        raw[offset] ^= 1  # the header checksum no longer matches
        damaged = gzip.compress(bytes(raw))
        (self.fx.root / "demo" / "zz.txt").unlink()
        self.fx.commit("delete zz.txt, same version")
        rc, out = self.fx.run("HEAD", lambda n, v: damaged)
        self.assertEqual(rc, 2, out)
        self.assertIn("archive is not complete", out)

    def test_archive_cut_at_a_member_boundary_is_not_green(self) -> None:
        # #415: the same short parse can happen with no damaged header at all.
        # Cutting the stream exactly where the last member's data ends leaves
        # tarfile a clean end-of-file and an empty tail, so only the missing
        # end-of-archive blocks give it away.
        self.fx.write("zz.txt", "zz\n")
        self.fx.commit("add zz.txt")
        raw = gzip.decompress(self.fx.package())
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
            last = tf.getmembers()[-1]
        cut = last.offset + 512 + -(-last.size // 512) * 512  # end of its data
        rc, out = self.fx.run("HEAD", lambda n, v: gzip.compress(raw[:cut]))
        self.assertEqual(rc, 2, out)
        self.assertIn("archive is not complete", out)

    def test_zeroed_member_header_is_not_green(self) -> None:
        # GPT review on #416 (critical): zero the final member's 512-byte
        # header, so tarfile takes it for the end-of-archive marker. The zero
        # payload and padding behind it pass the tail check, and the publication
        # commit is still an ancestor, so this used to report ok (exit 0) for a
        # file that is gone from the tree. Only the checksum crates.io
        # registered for the download sees it.
        self.fx.write("zz.txt", "\0" * 512)  # all-zero payload, all the way to the tail
        self.fx.commit("add zz.txt")
        published = self.fx.package()
        raw = bytearray(gzip.decompress(published))
        with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as tf:
            offset = tf.getmember("demo-0.1.0/zz.txt").offset
        raw[offset:offset + 512] = b"\0" * 512
        damaged = gzip.compress(bytes(raw))
        (self.fx.root / "demo" / "zz.txt").unlink()
        self.fx.commit("delete zz.txt, same version")
        with self._fake_crates_io(200, damaged, cksum=hashlib.sha256(published).hexdigest()):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("does not match the sha256 crates.io registered", out)
        self.assertNotIn("ok          demo", out)

    VCS = "demo-0.1.0/.cargo_vcs_info.json"

    def test_zeroed_tree_member_header_is_not_green(self) -> None:
        # GPT review on #416 (critical, tree side): the same zeroed-header trick
        # against the archive *we* build. The registered checksum guards the
        # published side only, so the tree archive is judged by the length of its
        # end-of-archive blocks and by the file list `cargo package --list`
        # names, both independently of its headers.
        published = self.fx.package()
        self.fx.write("zz.txt", "\0" * 512)
        self.fx.commit("add shipped file without bumping version")
        real = pi.package_tree

        def corrupted(*args):
            raw = bytearray(gzip.decompress(real(*args)))
            with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as tf:
                offset = tf.getmember("demo-0.1.0/zz.txt").offset
            raw[offset:offset + 512] = b"\0" * 512
            return gzip.compress(bytes(raw))

        with self._fake_crates_io(200, published):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 1, out)  # control: the added file is seen
        self.assertIn("only in", out)
        with self._fake_crates_io(200, published), \
                mock.patch.object(pi, "package_tree", corrupted):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 2, out)
        self.assertIn("archive is not complete", out)  # the pinned end blocks
        self.assertNotIn("ok          demo", out)

    def _repacked(self, vcs: dict | None) -> bytes:
        """The published blob with its .cargo_vcs_info.json replaced (None: dropped)."""
        body = None if vcs is None else json.dumps(vcs).encode()
        return repack(self.fx.package(), {self.VCS: body})

    def test_tree_archive_missing_a_member_is_not_green(self) -> None:
        # #416, in the form the tar reader cannot see: a structurally complete
        # tree archive (both end-of-archive blocks, nothing behind them) that
        # simply lacks a member. Only the independent `cargo package --list`
        # comparison catches it.
        blob = self.fx.package()
        real = pi.package_tree

        def shorter(*args):
            return repack(real(*args), {"demo-0.1.0/src/lib.rs": None})

        with mock.patch.object(pi, "package_tree", shorter):
            rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("not the file list Cargo shipped", out)

    def test_long_member_name_checks_the_payload_offset(self) -> None:
        # GPT review on #416 (round 3): a name longer than 100 bytes puts a pax
        # extension in front of the header, and tarfile's `offset` points at
        # that extension while `offset_data` points at the payload. Using
        # `offset + 512` reads the extension as payload padding and rejects an
        # unchanged archive.
        self.fx.write("a" * 120 + ".txt", "x\n")
        self.fx.commit("long file name")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)

    def test_tree_member_size_field_lie_is_not_green(self) -> None:
        # GPT review on #416 (critical, third round): flip one bit of a tree
        # member's size field ('3' -> '2') and one of its UID ('0' -> '1'); the
        # two cancel in the header checksum, so tarfile accepts the header, reads
        # 'ab' out of 'abc' and takes the remaining 'c' for padding. Both sizes
        # sit in one 512-byte block, so the end blocks and the `cargo package
        # --list` comparison are unchanged, and the short member matched the
        # published two-byte file: only the bytes on disk disagree.
        self.fx.write("zz.txt", "ab")
        self.fx.commit("published two-byte file")
        published = self.fx.package()
        self.fx.write("zz.txt", "abc")
        self.fx.commit("changed content without version bump")
        real = pi.package_tree

        def corrupted(*args):
            raw = bytearray(gzip.decompress(real(*args)))
            with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as tf:
                o = tf.getmember("demo-0.1.0/zz.txt").offset
            assert raw[o + 134] == ord("3") and raw[o + 114] == ord("0")
            raw[o + 134] ^= 1  # size 3 -> 2
            raw[o + 114] ^= 1  # UID 0 -> 1; the two cancel in the checksum
            return gzip.compress(bytes(raw))

        rc, out = self.fx.run("HEAD", lambda n, v: published)
        self.assertEqual(rc, 1, out)  # control: the changed byte is seen
        self.assertIn("differs: zz.txt", out)
        with mock.patch.object(pi, "package_tree", corrupted):
            rc, out = self.fx.run("HEAD", lambda n, v: published)
        self.assertEqual(rc, 2, out)
        # The nonzero leftover is caught by the padding check first; the NUL
        # variant below is the one only the bytes on disk catch.
        self.assertIn("the size field does not describe what the archive wrote", out)
        self.assertNotIn("ok          demo", out)

    def test_tree_member_size_lie_over_a_zero_byte_is_not_green(self) -> None:
        # The same edit, but the byte the shortened size drops is a NUL: the
        # leftover looks exactly like the padding, so the padding check alone
        # stays green here and only the bytes on disk disagree.
        self.fx.write("zz.txt", "ab")
        self.fx.commit("published two-byte file")
        published = self.fx.package()
        self.fx.write("zz.txt", "ab\0")
        self.fx.commit("grew by one NUL byte without a version bump")
        real = pi.package_tree

        def corrupted(*args):
            raw = bytearray(gzip.decompress(real(*args)))
            with tarfile.open(fileobj=io.BytesIO(bytes(raw)), mode="r:") as tf:
                o = tf.getmember("demo-0.1.0/zz.txt").offset
            assert raw[o + 134] == ord("3") and raw[o + 114] == ord("0")
            raw[o + 134] ^= 1  # size 3 -> 2
            raw[o + 114] ^= 1  # UID 0 -> 1; the two cancel in the checksum
            return gzip.compress(bytes(raw))

        rc, out = self.fx.run("HEAD", lambda n, v: published)
        self.assertEqual(rc, 1, out)  # control: the added NUL is seen
        self.assertIn("differs: zz.txt", out)
        with mock.patch.object(pi, "package_tree", corrupted):
            rc, out = self.fx.run("HEAD", lambda n, v: published)
        self.assertEqual(rc, 2, out)
        self.assertIn("the member does not carry the bytes on disk", out)

    def test_unchanged_file_with_a_carriage_return_in_its_name_is_ok(self) -> None:
        # GPT review on #416 (post-review): text-mode stdout turns a CR in a
        # name into an LF, which would report an unchanged file as changed.
        self.fx.write("notes\rmore", "x\n")
        self.fx.commit("file name with a carriage return")
        blob = self.fx.package()
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 0, out)
        self.assertIn("ok          demo 0.1.0", out)
    def test_size_field_edited_with_the_checksum_intact_is_not_green(self) -> None:
        # GPT review on #416 (21:10Z): move one bit between the size and the UID
        # fields of a member header of the TREE-built archive, so the header
        # checksum still matches while tarfile reads a one-byte-shorter member.
        # The real content leftover byte stays in the padding, which no header
        # or end-of-archive check can see, so the padding has to be checked.
        self.fx.write("zz.txt", "ab")
        self.fx.commit("published two-byte file")
        published = self.fx.package()
        self.fx.write("zz.txt", "abc")
        self.fx.commit("changed content without a version bump")
        real_package = pi.package_tree

        def corrupted(*args):
            raw = bytearray(gzip.decompress(real_package(*args)))
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
                o = tf.getmember("demo-0.1.0/zz.txt").offset
            self.assertEqual(raw[o + 134], ord("3"))
            self.assertEqual(raw[o + 114], ord("0"))
            raw[o + 134] ^= 1  # size 3 -> 2
            raw[o + 114] ^= 1  # UID 0 -> 1; the checksum is unchanged
            return gzip.compress(raw)

        with mock.patch.object(pi, "package_tree", corrupted):
            rc, out = self.fx.run("HEAD", lambda n, v: published)
        self.assertEqual(rc, 2, out)
        self.assertIn("zero padding", out)

    def test_published_from_a_dirty_worktree_is_red(self) -> None:
        # #415: cargo records the worktree state it packaged. dirty=true means
        # the version names no committed content at all.
        sha = sh(self.fx.root, "git", "rev-parse", "HEAD").strip()
        blob = self._repacked({"git": {"sha1": sha, "dirty": True}})
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("dirty", out)

    def test_published_from_a_commit_outside_the_history_is_red(self) -> None:
        # #415: a commit that exists but is not an ancestor of the ref we check
        # (a side branch, a rewritten history) is not what this version claims.
        dangling = sh(self.fx.root, "git", "commit-tree", "HEAD^{tree}", "-m", "elsewhere").strip()
        blob = self._repacked({"git": {"sha1": dangling}})
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 1, out)
        self.assertIn("not an ancestor", out)

    def test_unknown_recorded_commit_is_not_green(self) -> None:
        # #415: a clone without that commit cannot tell which commit published
        # an archive. That is "could not check", never "no differences".
        blob = self._repacked({"git": {"sha1": "0" * 40}})
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("not in this repository", out)

    def test_recorded_sha_must_be_a_sha(self) -> None:
        # #415: a 40-character revision expression is not a commit id; `HEAD~34`
        # can resolve here, so length alone is no validation.
        blob = self._repacked({"git": {"sha1": "HEAD" + "~" * 36}})
        rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("no usable commit", out)

    def test_shallow_clone_is_reported_as_not_checked(self) -> None:
        # #415: with the parent links cut, "not an ancestor" is not an answer
        # either, so a shallow clone says so instead of blaming the archive.
        blob = self.fx.package()
        clone = pathlib.Path(self._tmp.name) / "shallow"
        self.fx.write("b.txt", "b\n")
        self.fx.commit("second commit")
        subprocess.run(["git", "clone", "-q", "--depth", "1",
                        f"file://{self.fx.root}", str(clone)], check=True)
        out = io.StringIO()
        with mock.patch.object(pi, "published", lambda n, v: blob), \
                mock.patch("sys.argv", ["published_identity.py", str(clone), "--ref", "HEAD"]), \
                contextlib.redirect_stdout(out):
            rc = pi.main()
        self.assertEqual(rc, 2, out.getvalue())
        self.assertIn("shallow clone", out.getvalue())

    def test_terminated_ancestry_check_is_not_green(self) -> None:
        # GPT review on #416: a negative return code is how Python reports a
        # signal (SIGTERM → -15). That is no ancestry answer, so returning "no
        # differences" for it is a false green.
        blob = self.fx.package()
        with self._patch_run("--is-ancestor", (-15, "", "")):
            rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("could not place commit", out)

    def test_failed_shallow_query_is_not_green(self) -> None:
        # GPT review on #416: the shallow-history query failing means the clone's
        # completeness is unknown, which is "could not check", not "no
        # differences".
        blob = self.fx.package()
        with self._patch_run("--is-shallow-repository", (128, "", "fatal: injected git error")):
            rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("could not tell whether this clone is shallow", out)

    def test_unrecognised_shallow_answer_is_not_green(self) -> None:
        # GPT review on #416 (follow-up): exit 0 with output that is neither
        # `true` nor `false` answers nothing either, even when the files match.
        blob = self.fx.package()
        with self._patch_run("--is-shallow-repository", (0, "", "")):
            rc, out = self.fx.run("HEAD", lambda n, v: blob)
        self.assertEqual(rc, 2, out)
        self.assertIn("does not say whether commit", out)

    def test_archive_without_vcs_info_is_not_green(self) -> None:
        # #415: no recorded commit means no provenance evidence; report that
        # rather than comparing files and calling it ok.
        rc, out = self.fx.run("HEAD", lambda n, v: self._repacked(None))
        self.assertEqual(rc, 2, out)
        self.assertIn("no .cargo_vcs_info.json", out)

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

    @staticmethod
    def _opener(api_status: int, crate_blob: bytes | None = None, cksum: str | None = None):
        """A urlopen-compatible opener: the API URL answers `api_status` with the
        checksum crates.io registered for `crate_blob` (override with `cksum`),
        the static download answers 200 with `crate_blob`. Non-2xx statuses are
        raised as HTTPError, exactly as urllib does, so fetch()'s HTTPError
        branch runs."""
        import urllib.error

        class Resp(io.BytesIO):
            def __init__(self, body: bytes) -> None:
                super().__init__(body)
                self.status = 200

        registered = cksum or (
            hashlib.sha256(crate_blob).hexdigest() if crate_blob is not None else "0" * 64)

        def urlopen(req, *args, **kwargs):
            url = req.full_url
            if "/api/v1/crates/" in url:
                if api_status != 200:
                    raise urllib.error.HTTPError(url, api_status, "status", {}, None)
                return Resp(json.dumps({"version": {"checksum": registered}}).encode())
            if "static.crates.io" in url and crate_blob is not None:
                return Resp(crate_blob)
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)

        return urlopen

    def _fake_crates_io(self, api_status: int, crate_blob: bytes | None = None,
                        cksum: str | None = None):
        """For main()-level runs: the default opener (urllib.request.urlopen,
        looked up at call time) is replaced by `_opener(...)`."""
        return mock.patch.object(pi.urllib.request, "urlopen",
                                 self._opener(api_status, crate_blob, cksum))

    def test_api_503_through_urlopen_is_not_green(self) -> None:
        # GPT review on #411 (round 6): an HTTPError 503 from the API must be
        # exit 2. Mutating fetch()'s HTTPError branch to `return 404, b""`
        # turns this red; mocking fetch() itself could not see that.
        with self._fake_crates_io(503):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 2, out)
        self.assertNotIn("unpublished", out)
        self.assertNotIn("MISMATCH", out)

    def test_api_404_through_urlopen_is_unpublished(self) -> None:
        with self._fake_crates_io(404):
            rc, out = self.fx.run("HEAD", real_network=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("unpublished demo 0.1.0", out)
        self.assertNotIn("MISMATCH", out)
        self.assertNotIn("ok          demo", out)

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
        self.assertIn("MISMATCH    demo 0.1.0", out)
        self.assertIn("differs: src/lib.rs", out)
        self.assertNotIn("unpublished", out)

    def test_injected_opener_503_from_published_is_a_check_error(self) -> None:
        # The opener is injected into published() directly; HTTPError(503) is
        # raised by the opener itself, not by a mocked fetch().
        with self.assertRaises(pi.CheckError):
            pi.published("demo", "0.1.0", opener=self._opener(503))

    def test_injected_opener_404_from_published_is_unpublished(self) -> None:
        self.assertIsNone(pi.published("demo", "0.1.0", opener=self._opener(404)))

    def test_injected_opener_200_from_published_returns_the_archive(self) -> None:
        blob = self.fx.package()
        self.assertEqual(pi.published("demo", "0.1.0", opener=self._opener(200, blob)), blob)

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

    def _patch_run(self, match: str, result: tuple[int, str, str]):
        """pi.run with the command whose argv contains `match` answering
        `result`; every other command still runs for real."""
        real = pi.run

        def fake(cmd, cwd=None, timeout=120):
            if any(match in part for part in cmd):
                return result
            return real(cmd, cwd=cwd, timeout=timeout)

        return mock.patch.object(pi, "run", fake)

    def test_only_404_means_unpublished(self) -> None:
        with mock.patch.object(pi, "fetch", lambda url, opener=None: (404, b"")):
            self.assertIsNone(pi.published("demo", "0.1.0"))
        with mock.patch.object(pi, "fetch", lambda url, opener=None: (403, b"")):
            with self.assertRaises(pi.CheckError):
                pi.published("demo", "0.1.0")

    def test_unknown_crate_name_is_not_green(self) -> None:
        out = io.StringIO()
        argv = ["published_identity.py", str(self.fx.root), "--crate", "nope"]
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(out):
            self.assertEqual(pi.main(), 2, out.getvalue())



class MutationTest(unittest.TestCase):
    """Each mutation is applied to a copy of the script, and the targeted tests
    are run against that copy in a subprocess; they must fail. The transport
    entries pin the round-6 review on #411, the others #415 and #416."""

    MUTATIONS = {
        # Comparison disabled after a successful status check.
        "none_after_api_200": (
            '    code, blob = fetch(f"https://static.crates.io',
            '    return None\n    code, blob = fetch(f"https://static.crates.io',
            ["PublishedIdentityTest.test_published_archive_with_different_source_through_urlopen_is_red",
             "PublishedIdentityTest.test_injected_opener_200_from_published_returns_the_archive"],
        ),
        # `return 404` in place of the HTTP error code.
        "http_error_as_404": (
            '        return e.code, b""',
            '        return 404, b""',
            ["PublishedIdentityTest.test_api_503_through_urlopen_is_not_green",
             "PublishedIdentityTest.test_injected_opener_503_from_published_is_a_check_error"],
        ),
        # The completeness check reduced to "nothing follows the last member",
        # which a stream cut at a member boundary also satisfies (#415).
        "tail_only_completeness": (
            "    if len(raw) - end != 1024 or any(raw[end:]):",
            "    if any(raw[end:]):",
            ["PublishedIdentityTest.test_archive_cut_at_a_member_boundary_is_not_green"],
        ),
        # The registry checksum not compared at all: the zeroed-header archive
        # reads as complete and shorter, and goes green (#416).
        "checksum_not_compared": (
            "    if got != registered.lower():",
            "    if False and got != registered.lower():",
            ["PublishedIdentityTest.test_zeroed_member_header_is_not_green"],
        ),
        # A terminated ancestry check (rc -15) read as "not an ancestor", which
        # is what `if rc == 1` alone would accept (#416).
        "terminated_ancestry_accepted": (
            "    if rc != 1:  # a negative rc is a signal (SIGTERM); >1 is a git error",
            "    if False:",
            ["PublishedIdentityTest.test_terminated_ancestry_check_is_not_green"],
        ),
        # A failing shallow-history query ignored (#416).
        "failed_shallow_query_ignored": (
            "    rc, out, _ = run([\"git\", \"rev-parse\", \"--is-shallow-repository\"], cwd=repo)\n    if rc != 0:",
            "    rc, out, _ = run([\"git\", \"rev-parse\", \"--is-shallow-repository\"], cwd=repo)\n    if False:",
            ["PublishedIdentityTest.test_failed_shallow_query_is_not_green"],
        ),
        # Any answer other than true/false accepted as "not shallow" (#416).
        "shallow_answer_not_checked": (
            '    if out.strip() != "false":',
            "    if False:",
            ["PublishedIdentityTest.test_unrecognised_shallow_answer_is_not_green"],
        ),
        # The tree archive's member list not compared with `cargo package
        # --list`: a complete archive that lost a member goes green (#416,
        # tree side).
        "tree_member_list_not_checked": (
            "    if listed is not None and",
            "    if False and",
            ["PublishedIdentityTest.test_tree_archive_missing_a_member_is_not_green"],
        ),
        # The tree archive's member bytes not compared with the file on disk: a
        # member shortened by a size-field edit that keeps the header checksum
        # then goes green against a published file of that shorter length
        # (#416, tree side).
        "tree_member_bytes_not_checked": (
            "    if on_disk != body:",
            "    if False:",
            ["PublishedIdentityTest.test_tree_member_size_lie_over_a_zero_byte_is_not_green"],
        ),
        # The end-of-archive check back to "something is there": a zeroed last
        # member header then leaves its own header and payload behind two
        # blocks that are already zeros (#416, tree side).
        "end_blocks_not_pinned": (
            "    if len(raw) - end != 1024 or any(raw[end:]):",
            "    if len(raw) - end < 1024 or any(raw[end:]):",
            ["PublishedIdentityTest.test_zeroed_tree_member_header_is_not_green"],
        ),
    }

    def test_each_mutation_turns_its_tests_red(self) -> None:
        src = (HERE / "published_identity.py").read_text()
        for name, (old, new, tests) in self.MUTATIONS.items():
            with self.subTest(mutation=name):
                self.assertEqual(src.count(old), 1, f"{name}: mutation site moved")
                with tempfile.TemporaryDirectory() as d:
                    (pathlib.Path(d) / "published_identity.py").write_text(src.replace(old, new))
                    (pathlib.Path(d) / "test-published-identity.py").write_text(
                        pathlib.Path(__file__).read_text())
                    for test in tests:
                        r = subprocess.run([sys.executable, f"{d}/test-published-identity.py", test],
                                           capture_output=True, text=True)
                        self.assertNotEqual(r.returncode, 0,
                                            f"{name}: {test} stayed green\n{r.stderr[-500:]}")

if __name__ == "__main__":
    unittest.main()
