#!/usr/bin/env python3
"""Check that a version number identifies one body of content.

On 2026-11-08 this invariant broke twice in the same repository:

  strided-basic 0.12.0   published from 97e6e41 (add / mul / fma / axpy),
                         then main changed to add_acc / mul_acc ... while the
                         manifest still said 0.12.0.
  strided-view  0.7.1    published with wrapping isize arithmetic, while main
                         said 0.7.1 and panicked instead.

Neither is a normal "forgot to bump" bug: in both cases `cargo update` gives a
downstream crate different behaviour for the same version string, and no test
in the workspace can see it, because the workspace builds against paths. The
only thing that can see it is a comparison of the published artefact with the
tree that claims that version. That is this script.

Usage:
  published_identity.py <repo-path> [--ref REF] [--crate NAME ...]

Exit codes: 0 = every crate's declared version is either unpublished or
matches the published artefact, 1 = at least one mismatch, 2 = could not check
(git, cargo, network or crates.io failure; never reported as green).

Caveats, stated rather than hidden:
  * The tree side is the .crate `cargo package --no-verify` builds from the
    checked-out commit, so every packaging rule is Cargo's own. Packaging
    resolves dependencies on crates.io; a published crate whose tree needs an
    unpublished dependency version cannot be packaged and exits 2.
  * Cargo.toml, Cargo.lock and .cargo_vcs_info.json are excluded from the
    byte comparison: Cargo generates or rewrites them at package time.
    Cargo.toml.orig, the crate's manifest exactly as written, IS compared, so a
    same-version change to features, dependencies or build settings in the
    crate's own manifest is a MISMATCH.
  * .cargo_vcs_info.json is not compared, but it IS read: it names the commit
    the archive was built from. A version published from a dirty worktree, or
    from a commit that is not an ancestor of the checked ref, is a MISMATCH;
    an archive that carries no such record, or names a commit this clone does
    not have, is exit 2 (nothing was verified). The ancestry answer is only
    accepted from `git merge-base --is-ancestor` exit 0 or 1; a git that fails
    to answer (including a terminated process) is exit 2, and so is a failure
    to say whether the clone is shallow.
  * The downloaded .crate is checked against the sha256 crates.io registered
    for that version before it is parsed, so a corrupt download is exit 2
    whatever a tar reader makes of it. A .crate whose final member header is
    zeroed reads as a complete, SHORTER archive (tarfile takes the zero header
    for the end-of-archive marker, and the zero payload and padding behind it
    pass a tail-is-zero check), which is the #415 false green; on the published
    side only the registered checksum sees that (the tree side, below, is caught
    by its pinned end blocks, its file list and its bytes on disk instead).
  * Both archives must parse completely: the parse must be followed by exactly
    the 1024 bytes of the two end-of-archive zero blocks (tar-rs writes those
    and no record padding behind them; a writer that padded further would fail
    loudly rather than silently). tarfile silently ends the member list at a
    damaged header after the first block, so a damaged .crate reads as a SHORTER
    one and a shipped file that is gone from the tree compares as absent on both
    sides (#415). A stream cut off exactly at a member boundary leaves nothing
    behind the last member. A zeroed last header leaves the two end blocks *and*
    that header and its payload behind, so an all-zero payload does not hide it
    (#416). Either way the run is exit 2.
  * The tree side does not rest on the tar reader for its member list: `cargo
    package --list` names the files Cargo ships, and the tree archive's member
    list must be that same text, in the same order (Cargo writes the tar in that
    order; a Cargo that listed in another order would fail the run loudly, not
    silently). Names are compared as bytes, so a CR or a name that is not UTF-8
    is not rewritten on the way. That catches a structurally valid archive that
    simply lacks a member. The published side has no such list — its tree is not
    checked out here — and rests on the registered checksum instead. Residual
    limit: the list is newline-delimited, so a name containing a newline is
    indistinguishable from that name split in two. Damage cannot split a header,
    only end the list or shorten a member (which the two caveats below catch),
    and the archive is built here, so that needs a replaced archive rather than a
    corrupted one.
  * A header edited so its checksum still matches (one bit moved between the
    size and UID fields) reads as a shorter member with no error from any of
    the checks above or from the file list. The tar format pads a payload to a
    512-byte boundary with zeros, so the leftover bytes of the real content can
    only sit in that padding, which is checked for every member of both
    archives (#416 review). Residue: a member whose last bytes are zeros loses
    nothing visible when such an edit drops one of them, which is why the next
    caveat does not rest on the padding alone.
  * Nor does the tree side rest on the tar reader for a member's content: the
    bytes read out of the archive are compared with the bytes of the file Cargo
    archived, on disk (a readme / license-file from outside the crate is read
    where the manifest points, since Cargo archives it at the package root under
    its own name). A member list and a set of end blocks say nothing about a
    member's length: a header whose size field is edited can still pass its own
    checksum — flip a bit in the size and one in the UID and the two cancel — so
    tarfile reads the member shorter and the padding swallows the rest (#416).
    A member whose bytes are not the file's bytes, or which names no file on
    disk to compare with, is exit 2.
  * Manifest comparison covers Cargo.toml.orig only; workspace-level
    inheritance is out of scope. A field written as `{ workspace = true }` (or
    `x.workspace = true`) reads the same in Cargo.toml.orig whatever the
    workspace root says, so a change made only in the root Cargo.toml's
    `[workspace.package]` / `[workspace.dependencies]` is not seen here.
  * The comparison is byte-for-byte: a comment-only change under the same
    version is a MISMATCH. That is intended — the published source is what
    a user reads and builds.
  * Crates whose declared version is not on crates.io yet are reported as
    `unpublished` and do not fail the run — that is a release in progress.
  * A crate with `publish = false` is skipped.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.error
import urllib.request

UA = "tensor4all-published-identity/1.0 (maintainer check)"
# Both sides are .crate archives built by Cargo (the published one, and one
# built from the tree with `cargo package`), so the file set is Cargo's. Only
# the files Cargo generates or rewrites at package time are left out.
# Cargo.toml.orig is NOT in this set: it is the author's manifest verbatim.
# VCS_INFO is read for provenance instead of being compared (see provenance()).
VCS_INFO = ".cargo_vcs_info.json"
SKIP_ALWAYS = {VCS_INFO, "Cargo.lock", "Cargo.toml"}


class CheckError(Exception):
    """The check itself could not run (network, crates.io, git); exit 2, never green."""


def run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """(returncode, stdout, stderr). The streams are kept apart: a warning on
    stderr must never end up inside JSON or a path read from stdout."""
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def checkout_ref(repo: str, ref: str) -> tuple[str, str | None]:
    """Materialise `ref` in a throwaway worktree.

    Reading the version from the working tree while reading the files from `ref`
    produced a false MISMATCH the first time this ran against a pull request
    (working tree said 0.7.0, the branch said 0.7.1). Both have to come from the
    same commit, and `cargo package` also needs the real manifest, so the ref is
    checked out rather than patched up in place.
    """
    tmp = tempfile.mkdtemp(prefix="published-identity-")
    wt = pathlib.Path(tmp) / "wt"
    rc, _, _ = run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=repo, timeout=300)
    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return repo, None
    return str(wt), tmp


def workspace_crates(repo: str) -> list[dict]:
    """The publishable crates, with the crate root and the declared readme /
    license-file: those say where each archived member's bytes live on disk
    (see member_bytes())."""
    rc, out, err = run(["cargo", "metadata", "--no-deps", "--format-version", "1",
                        "--manifest-path", f"{repo}/Cargo.toml"], cwd=repo)
    if rc != 0:
        raise CheckError(f"cargo metadata failed: {err[:300]}")
    meta = json.loads(out)
    crates = []
    for p in meta["packages"]:
        if p.get("publish") == []:
            continue
        crates.append({"name": p["name"], "version": p["version"],
                       "root": str(pathlib.Path(p["manifest_path"]).parent),
                       "readme": p.get("readme"), "license_file": p.get("license_file")})
    return crates


def fetch(url: str, opener=None) -> tuple[int, bytes]:
    """(status, body). `opener` defaults to urllib.request.urlopen, looked up
    at call time; tests inject one that raises HTTPError like urllib does."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    open_url = opener or urllib.request.urlopen
    try:
        with open_url(req, timeout=60) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # DNS, TLS, timeout: not evidence of anything
        raise CheckError(f"GET {url}: {e}") from e


def published(name: str, ver: str, opener=None) -> bytes | None:
    """The published .crate, or None when crates.io has no such version.

    Only a 404 from the API counts as "not published". Any other failure is a
    failure to check: treating a network error as `unpublished` would turn an
    outage into a green run.

    The download is checked against the sha256 crates.io registered for it: the
    only integrity claim here that does not come from a parser of the file
    itself (see the caveat at the top of this file for the zeroed-header case
    that no tar-level check can see).
    """
    code, body = fetch(f"https://crates.io/api/v1/crates/{name}/{ver}", opener)
    if code == 404:
        return None
    if code != 200:
        raise CheckError(f"crates.io API returned {code} for {name} {ver}")
    try:
        registered = json.loads(body)["version"]["checksum"]
        if not isinstance(registered, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", registered):
            raise ValueError(registered)
    except (ValueError, KeyError, TypeError) as e:
        raise CheckError(f"crates.io registered no sha256 for {name} {ver}: {e!r}") from e
    code, blob = fetch(f"https://static.crates.io/crates/{name}/{name}-{ver}.crate", opener)
    if code != 200 or not blob:
        raise CheckError(f"download of {name}-{ver}.crate returned {code}")
    got = hashlib.sha256(blob).hexdigest()
    if got != registered.lower():
        raise CheckError(f"{name}-{ver}.crate does not match the sha256 crates.io registered"
                         f" (registered {registered.lower()}, downloaded {got})")
    return blob


def member_bytes(crate: dict, rel: str, body: bytes) -> None:
    """The tree archive's member `rel` must be the file Cargo archived, byte for
    byte. Neither the member list nor the end blocks see a member's length, and
    a size field edited with a compensating UID bit keeps the header checksum,
    so tarfile reads the member shorter and the padding swallows the rest: only
    an independent copy of the bytes, the file on disk, catches that (#416).
    """
    root = pathlib.Path(crate["root"])
    # Cargo.toml.orig is the crate's manifest verbatim.
    source = root / ("Cargo.toml" if rel == "Cargo.toml.orig" else rel)
    if not source.is_file():
        # Cargo archives a readme / license-file from outside the crate at the
        # package root, under its own file name.
        for key in ("readme", "license_file"):
            declared = crate.get(key)
            if declared and pathlib.Path(declared).name == rel:
                source = root / declared
    try:
        on_disk = source.read_bytes()
    except OSError as e:
        raise CheckError(f"the tree archive has {rel}, and there is no file on disk"
                         f" to compare it with: {e}") from e
    if on_disk != body:
        raise CheckError(
            f"the tree archive's {rel} is {len(body)} byte(s) and is not {source}"
            f" ({len(on_disk)} byte(s)): the member does not carry the bytes on disk")


def crate_archive(blob: bytes, listed: bytes | None = None,
                  crate: dict | None = None) -> tuple[dict[str, str], dict | None]:
    """(path-inside-crate -> sha256 for the files worth comparing, the archive's
    .cargo_vcs_info.json or None).

    `listed` is that same archive's file list from `cargo package --list`, in
    bytes, and `crate` the crate the tree side was packaged from; passing both
    means a structurally valid archive cannot lose a member or shorten one
    unnoticed (see the caveats at the top of this file).
    """
    # Decompress the whole stream first: tarfile over a gzip stream stops at
    # the tar end marker and never checks the gzip trailer, so a truncated or
    # CRC-damaged download would otherwise read as valid. gzip.decompress
    # raises (BadGzipFile / EOFError), which main() turns into exit 2.
    raw = gzip.decompress(blob)
    out: dict[str, str] = {}
    vcs: dict | None = None
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        try:
            entries = list(tf)  # a damaged header can raise ReadError here
        except tarfile.TarError as e:
            raise CheckError(f"archive is not readable: {e}") from e
        # ...and tarfile ENDS the list silently at other damaged headers (an
        # invalid checksum after the first block), which would let a file that
        # shipped and is now gone compare as absent on both sides (#415). A
        # complete read is proved only by what follows the last member, below.
        end = tf.offset
        names: list[str] = []
        for m in entries:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            names.append(rel)
            f = tf.extractfile(m)
            if f is None:
                continue
            body = f.read()
            # The tar format pads the payload to a 512-byte boundary with
            # zeros, and tarfile truncates at the size in the header. A
            # checksum-preserving edit of that field reads as a shorter member
            # with no error from the parse, the header checksum or the
            # end-of-archive check (#416 review), and the real content leftover
            # bytes stay in the padding, so a nonzero leftover shows here; a
            # member that ends in zeros shows nothing, which is why member_bytes()
            # does not rest on the padding.
            pad_start = m.offset + 512 + m.size
            pad_end = m.offset + 512 + -(-m.size // 512) * 512
            if any(raw[pad_start:pad_end]):
                raise CheckError(
                    f"{m.name}: the bytes after the declared content are not"
                    " the zero padding a complete member ends with, so the"
                    " size field does not describe what the archive wrote")
            if rel == VCS_INFO:
                vcs = json.loads(body)
            elif rel not in SKIP_ALWAYS:
                out[rel] = hashlib.sha256(body).hexdigest()
                if crate is not None:
                    member_bytes(crate, rel, body)
    # Two zero blocks are what a complete tar ends with, and tar-rs writes
    # nothing behind them. A stream cut off exactly at a member boundary leaves
    # an empty tail, and a stream cut off at a damage point leaves the rest of
    # the archive there, so "nothing follows" is not completeness either. A
    # zeroed last header leaves the two end blocks plus that header and its
    # payload behind, which is why the length is pinned and not just the bytes.
    if len(raw) - end != 1024 or any(raw[end:]):
        raise CheckError(
            f"archive is not complete: {end} of {len(raw)} bytes parsed, and"
            f" the {len(raw) - end} bytes after them are not exactly the two"
            " end-of-archive zero blocks a complete tar ends with")
    if listed is not None and "".join(n + "\n" for n in names).encode(
            sys.getfilesystemencoding(), "surrogateescape") != listed:
        raise CheckError(
            "the tree archive is not the file list Cargo shipped (`cargo package"
            f" --list`): it parses as {len(names)} member(s), and they are not"
            " the files Cargo named")
    return out, vcs


def provenance(repo: str, ref: str, crate: dict, vcs: dict | None) -> list[str]:
    """What commit the published archive says it was built from.

    A second signal next to the file hashes, and a stronger one: it does not
    go through `cargo package`. A version published from a dirty worktree, or
    from a commit outside this history, does not identify our content; an
    archive that records no commit, or names one this clone does not have,
    cannot be checked and says so instead of comparing files and going green.
    """
    label = f"{crate['name']} {crate['version']}"
    if vcs is None:
        raise CheckError("the published archive has no .cargo_vcs_info.json, "
                         "so it records no commit to check")
    git = vcs.get("git") or {}
    if git.get("dirty"):
        return [f"{label}: published from a dirty worktree"
                " (.cargo_vcs_info.json says dirty=true)"]
    sha = git.get("sha1")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
        raise CheckError(f"the published archive names no usable commit: {sha!r}")
    rc, out, _ = run(["git", "rev-parse", "--is-shallow-repository"], cwd=repo)
    if rc != 0:
        raise CheckError(f"could not tell whether this clone is shallow, so commit"
                         f" {sha} cannot be placed in history"
                         f" (git rev-parse --is-shallow-repository exited {rc})")
    if out.strip() == "true":
        raise CheckError(f"this is a shallow clone, so commit {sha} cannot be placed"
                         " in history (check out the full history)")
    if out.strip() != "false":
        raise CheckError(f"git rev-parse --is-shallow-repository answered {out.strip()!r},"
                         f" which does not say whether commit {sha} can be placed in history")
    rc, _, err = run(["git", "cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo)
    if rc != 0:
        raise CheckError(f"commit {sha} recorded in the published archive is not in"
                         f" this repository: {err.strip()[:200]}")
    rc, _, _ = run(["git", "merge-base", "--is-ancestor", sha, ref], cwd=repo)
    if rc == 0:  # 0 = ancestor, 1 = not an ancestor; nothing else answers the question
        return []
    if rc != 1:  # a negative rc is a signal (SIGTERM); >1 is a git error
        raise CheckError(f"could not place commit {sha} in history"
                         f" (git merge-base --is-ancestor exited {rc})")
    return [f"{label}: published from {sha[:10]}, which is not an ancestor"
            f" of {ref[:10]}"]


def package_tree(repo: str, crate: dict, target_dir: str) -> bytes:
    """The .crate that `cargo package` builds from the checked-out tree.

    Comparing against Cargo's own archive, rather than re-deriving it from
    `cargo package --list` plus the source files, leaves every packaging rule
    to Cargo: include/exclude, symlinks (archived as their target's bytes), a
    `readme` / `license-file` outside the crate directory (archived at the
    package root under its declared name), and a collision between such a
    file and a crate-local one. Re-implementing those rules produced false
    results in review (#411).

    Dependencies are resolved against crates.io as for a real publish, so a
    tree that requires an unpublished dependency version cannot be packaged;
    that is reported as "could not check" (exit 2), not as green.
    """
    name, ver = crate["name"], crate["version"]
    rc, _, err = run(["cargo", "package", "-p", name, "--no-verify", "-q",
                      "--manifest-path", f"{repo}/Cargo.toml",
                      "--target-dir", target_dir], cwd=repo, timeout=600)
    if rc != 0:
        raise CheckError(f"cargo package failed for {name}: {err[:300]}")
    path = pathlib.Path(target_dir) / "package" / f"{name}-{ver}.crate"
    try:
        return path.read_bytes()
    except OSError as e:
        raise CheckError(f"cargo package produced no {path.name}: {e}") from e


def package_list(repo: str, crate: dict) -> bytes:
    """The files `cargo package` ships, one name per line, from Cargo itself but
    through a channel other than the archive's headers.

    The bytes agree, so this is no independent packaging implementation: it is
    the same list, read from stdout instead of from a tar a damaged header can
    end early. stdout is taken as bytes, because text mode would rewrite a CR
    in a name into an LF and make an unchanged file look changed.
    """
    name = crate["name"]
    r = subprocess.run(["cargo", "package", "--list", "-p", name, "--no-verify", "-q",
                        "--manifest-path", f"{repo}/Cargo.toml"],
                       capture_output=True, cwd=repo, timeout=600)
    if r.returncode != 0:
        raise CheckError(f"cargo package --list failed for {name}:"
                         f" {r.stderr.decode('utf-8', 'replace')[:300]}")
    return r.stdout


def check_crate(repo: str, ref: str, crate: dict, target_dir: str) -> tuple[str, list[str]]:
    name, ver = crate["name"], crate["version"]
    blob = published(name, ver)
    if blob is None:
        return "unpublished", [f"{name} {ver} is not on crates.io (release in progress?)"]
    # Both sides are complete archives; only the generated files in
    # SKIP_ALWAYS are left out. A file that shipped but is gone from the tree
    # (a deleted build.rs, say) shows as "only in published .crate".
    pub, vcs = crate_archive(blob)
    tree, _ = crate_archive(package_tree(repo, crate, target_dir),
                            package_list(repo, crate), crate)
    diffs: list[str] = []
    for rel in sorted(set(pub) | set(tree)):
        if rel not in pub:
            diffs.append(f"only in {ref}: {rel}")
        elif rel not in tree:
            diffs.append(f"only in published .crate: {rel}")
        elif pub[rel] != tree[rel]:
            diffs.append(f"differs: {rel}")
    diffs += provenance(repo, ref, crate, vcs)
    if diffs:
        return "MISMATCH", [f"{name} {ver}"] + [f"    {d}" for d in diffs[:20]] + (
            [f"    ... and {len(diffs) - 20} more"] if len(diffs) > 20 else [])
    return "ok", []


def _main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--crate", action="append", default=[])
    a = ap.parse_args()

    # Resolve once, in the caller's repository, and use the commit SHA from
    # here on. A relative revision such as `HEAD~1` would otherwise name a
    # different commit inside the temporary worktree.
    rc, out, err = run(["git", "rev-parse", "--verify", f"{a.ref}^{{commit}}"], cwd=a.repo)
    if rc != 0:
        print(f"cannot resolve {a.ref}: {err[:200]}")
        return 2
    ref = out.strip()
    tree, tmp = checkout_ref(a.repo, ref)
    if tmp is None:
        print(f"could not check out {ref}")
        return 2
    try:
        try:
            crates = workspace_crates(tree)
        except CheckError as e:
            print(f"could not list crates: {e}")
            return 2
        if a.crate:
            unknown = sorted(set(a.crate) - {c["name"] for c in crates})
            if unknown:
                print(f"not a publishable workspace crate: {', '.join(unknown)}")
                return 2
            crates = [c for c in crates if c["name"] in a.crate]
        if not crates:
            print("no publishable crate found; nothing was checked")
            return 2
        bad = 0
        for c in crates:
            try:
                verdict, lines = check_crate(tree, ref, c, os.path.join(tmp, "target"))
            except CheckError as e:
                print(f"could not check {c['name']} {c['version']}: {e}")
                return 2
            print(f"{verdict:11s} {c['name']} {c['version']}")
            for line in lines:
                print(line)
            if verdict == "MISMATCH":
                bad += 1
    finally:
        run(["git", "worktree", "remove", "--force", tree], cwd=a.repo, timeout=120)
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    print(f"{len(crates)} crate(s) checked at {ref}: {bad} mismatch(es)")
    return 1 if bad else 0


def main() -> int:
    """Exit 1 is reserved for a content MISMATCH. Anything that stops the check
    from running (missing cargo/git, a timeout, a malformed archive, bad JSON)
    is exit 2, so it can never be read as either green or a mismatch."""
    try:
        return _main()
    except Exception as e:  # noqa: BLE001 - the contract is "could not check"
        print(f"could not check: {type(e).__name__}: {e}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
