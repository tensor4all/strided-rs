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
  * Cargo.toml is excluded. `cargo package` rewrites it, so a byte comparison
    would always differ. The version string selects which .crate is compared,
    so it is not compared itself.
  * The comparison is byte-for-byte: a comment-only change under the same
    version is a MISMATCH. That is intended — the published source is what
    a user reads and builds.
  * Crates whose declared version is not on crates.io yet are reported as
    `unpublished` and do not fail the run — that is a release in progress.
  * A crate with `publish = false` is skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import tarfile
import urllib.error
import urllib.request

UA = "tensor4all-published-identity/1.0 (maintainer check)"
# Files worth comparing: what a user compiles. Manifest and lockfiles excluded.
# The file set is taken from `cargo package --list`, which honours each crate's
# `include` / `exclude`. Inventing a file set here produced false positives on
# files a crate deliberately does not ship (#402 removed exactly such a test).
SKIP_ALWAYS = {".cargo_vcs_info.json", "Cargo.lock", "Cargo.toml", "Cargo.toml.orig"}


class CheckError(Exception):
    """The check itself could not run (network, crates.io, git); exit 2, never green."""


def run(cmd: list[str], cwd: str | None = None, timeout: int = 120) -> tuple[int, str]:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr)


def checkout_ref(repo: str, ref: str) -> tuple[str, str | None]:
    """Materialise `ref` in a throwaway worktree.

    Reading the version from the working tree while reading the files from `ref`
    produced a false MISMATCH the first time this ran against a pull request
    (working tree said 0.7.0, the branch said 0.7.1). Both have to come from the
    same commit, and `cargo package --list` also needs the real manifest, so the
    ref is checked out rather than patched up in place.
    """
    tmp = tempfile.mkdtemp(prefix="published-identity-")
    wt = pathlib.Path(tmp) / "wt"
    rc, out = run(["git", "worktree", "add", "--detach", str(wt), ref], cwd=repo, timeout=300)
    if rc != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        return repo, None
    return str(wt), tmp


def workspace_crates(repo: str, ref: str) -> list[dict]:
    rc, out = run(["cargo", "metadata", "--no-deps", "--format-version", "1",
                   f"--manifest-path", f"{repo}/Cargo.toml"])
    if rc != 0:
        raise CheckError(f"cargo metadata failed: {out[:300]}")
    meta = json.loads(out)
    crates = []
    for p in meta["packages"]:
        if p.get("publish") == []:
            continue
        man = pathlib.Path(os.path.realpath(p["manifest_path"])).parent
        rel = man.relative_to(pathlib.Path(os.path.realpath(repo)))
        # Files that live outside the crate directory but ship inside the
        # .crate at the root under their own name (`readme = "../README.md"`,
        # `license-file = "../LICENSE"`). Map the packaged name to the source.
        relocated: dict[str, str] = {}
        for key in ("readme", "license_file"):
            v = p.get(key)
            if not v:
                continue
            src = pathlib.Path(os.path.realpath(man / v))
            if man not in src.parents:
                relocated[src.name] = str(src)
        crates.append({"name": p["name"], "version": p["version"], "dir": str(rel),
                       "relocated": relocated})
    return crates


def fetch(url: str) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as f:
            return f.status, f.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # DNS, TLS, timeout: not evidence of anything
        raise CheckError(f"GET {url}: {e}") from e


def published(name: str, ver: str) -> bytes | None:
    """The published .crate, or None when crates.io has no such version.

    Only a 404 from the API counts as "not published". Any other failure is a
    failure to check: treating a network error as `unpublished` would turn an
    outage into a green run.
    """
    code, _ = fetch(f"https://crates.io/api/v1/crates/{name}/{ver}")
    if code == 404:
        return None
    if code != 200:
        raise CheckError(f"crates.io API returned {code} for {name} {ver}")
    code, blob = fetch(f"https://static.crates.io/crates/{name}/{name}-{ver}.crate")
    if code != 200 or not blob:
        raise CheckError(f"download of {name}-{ver}.crate returned {code}")
    return blob


def crate_files(blob: bytes) -> dict[str, str]:
    """path-inside-crate -> sha256 for the files worth comparing."""
    out: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            if rel in SKIP_ALWAYS:
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            out[rel] = hashlib.sha256(f.read()).hexdigest()
    return out


def shipped_paths(repo: str, crate: dict) -> set[str]:
    """The exact files `cargo package` would ship, honouring include/exclude."""
    rc, out = run(["cargo", "package", "-p", crate["name"], "--list", "-q",
                   "--manifest-path", f"{repo}/Cargo.toml"], timeout=300)
    if rc != 0:
        raise CheckError(f"cargo package --list failed for {crate['name']}: {out[:300]}")
    return {line.strip() for line in out.splitlines()
            if line.strip() and line.strip() not in SKIP_ALWAYS}


def tree_files(repo: str, crate: dict, wanted: set[str]) -> dict[str, str]:
    """path-inside-crate -> sha256 of what `cargo package` would archive.

    `repo` is a clean checkout of the resolved commit, so the files are read
    from disk the way Cargo reads them: symlinks are followed (Cargo archives
    the target's bytes, not the link text) and a readme / license file outside
    the crate directory is read from where the manifest points. A shipped path
    that cannot be read fails the check rather than dropping out of it.
    """
    base = pathlib.Path(repo) / crate["dir"]
    files: dict[str, str] = {}
    for rel in wanted:
        src = pathlib.Path(crate["relocated"].get(rel, base / rel))
        try:
            files[rel] = hashlib.sha256(src.read_bytes()).hexdigest()
        except OSError as e:
            raise CheckError(f"cannot read shipped file {crate['name']}/{rel}: {e}") from e
    return files


def check_crate(repo: str, ref: str, crate: dict) -> tuple[str, list[str]]:
    name, ver, cdir = crate["name"], crate["version"], crate["dir"]
    blob = published(name, ver)
    if blob is None:
        return "unpublished", [f"{name} {ver} is not on crates.io (release in progress?)"]
    wanted = shipped_paths(repo, crate)
    # The published side is NOT filtered by `wanted`: a file that shipped in
    # the .crate but is gone from the tree (a deleted build.rs, say) changes
    # what users build, and must show up as "only in published .crate".
    pub = crate_files(blob)
    tree = tree_files(repo, crate, wanted)
    diffs: list[str] = []
    for rel in sorted(set(pub) | set(tree)):
        if rel not in pub:
            diffs.append(f"only in {ref}: {rel}")
        elif rel not in tree:
            diffs.append(f"only in published .crate: {rel}")
        elif pub[rel] != tree[rel]:
            diffs.append(f"differs: {rel}")
    if diffs:
        return "MISMATCH", [f"{name} {ver} ({cdir})"] + [f"    {d}" for d in diffs[:20]] + (
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
    rc, out = run(["git", "rev-parse", "--verify", f"{a.ref}^{{commit}}"], cwd=a.repo)
    if rc != 0:
        print(f"cannot resolve {a.ref}: {out[:200]}")
        return 2
    ref = out.strip()
    tree, tmp = checkout_ref(a.repo, ref)
    if tmp is None:
        print(f"could not check out {ref}")
        return 2
    try:
        try:
            crates = workspace_crates(tree, ref)
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
                verdict, lines = check_crate(tree, ref, c)
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
