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
  * Cargo.toml, Cargo.toml.orig, Cargo.lock and .cargo_vcs_info.json are
    excluded: Cargo generates or rewrites them at package time. The version
    string selects which .crate is compared, so it is not compared itself.
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
# Both sides are .crate archives built by Cargo (the published one, and one
# built from the tree with `cargo package`), so the file set is Cargo's. Only
# the files Cargo generates or rewrites at package time are left out.
SKIP_ALWAYS = {".cargo_vcs_info.json", "Cargo.lock", "Cargo.toml", "Cargo.toml.orig"}


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
    rc, out, err = run(["cargo", "metadata", "--no-deps", "--format-version", "1",
                        "--manifest-path", f"{repo}/Cargo.toml"], cwd=repo)
    if rc != 0:
        raise CheckError(f"cargo metadata failed: {err[:300]}")
    meta = json.loads(out)
    crates = []
    for p in meta["packages"]:
        if p.get("publish") == []:
            continue
        crates.append({"name": p["name"], "version": p["version"]})
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


def check_crate(repo: str, ref: str, crate: dict, target_dir: str) -> tuple[str, list[str]]:
    name, ver = crate["name"], crate["version"]
    blob = published(name, ver)
    if blob is None:
        return "unpublished", [f"{name} {ver} is not on crates.io (release in progress?)"]
    # Both sides are complete archives; only the generated files in
    # SKIP_ALWAYS are left out. A file that shipped but is gone from the tree
    # (a deleted build.rs, say) shows as "only in published .crate".
    pub = crate_files(blob)
    tree = crate_files(package_tree(repo, crate, target_dir))
    diffs: list[str] = []
    for rel in sorted(set(pub) | set(tree)):
        if rel not in pub:
            diffs.append(f"only in {ref}: {rel}")
        elif rel not in tree:
            diffs.append(f"only in published .crate: {rel}")
        elif pub[rel] != tree[rel]:
            diffs.append(f"differs: {rel}")
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
