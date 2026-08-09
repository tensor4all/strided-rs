# Publishing a strided-rs release

Publish only a commit that has already landed on `main`. The release tag must
identify that exact commit, and every crate must be published from a clean
checkout of the tag.

## 1. Prepare and validate `main`

Complete all version and dependency changes in a reviewed commit before the
release. Do not edit manifests while publishing: crates.io must receive the
same manifests recorded by the tag.

```bash
git switch main
git pull --ff-only origin main
cargo fmt --all -- --check
cargo test --workspace
```

Confirm all nine package archives contain their license and provenance files:

```bash
for crate in \
  strided-traits strided-view strided-perm strided-kernel \
  strided-einsum2 strided-opteinsum mdarray-opteinsum \
  ndarray-opteinsum strided-rs
do
  cargo package -p "$crate" --list
done
```

Every list must contain `LICENSE-APACHE`, `LICENSE-MIT`, and `NOTICE`.
`strided-perm` must also contain `THIRD-PARTY-LICENSES`.

## 2. Tag before publishing

Push the validated `main` commit before creating and pushing the release tag.
Replace `0.4.0` below for later releases.

```bash
set -euo pipefail

git push origin main
release_sha=$(git rev-parse HEAD)
test "$(git ls-remote origin refs/heads/main | cut -f1)" = "$release_sha"

# Wait for the required CI workflow triggered by this exact pushed commit.
run_id=
for attempt in {1..30}; do
  run_id=$(gh run list --workflow ci.yml --branch main --event push \
    --commit "$release_sha" --limit 1 --json databaseId --jq '.[0].databaseId // empty')
  test -z "$run_id" || break
  sleep 10
done
test -n "$run_id"
gh run watch "$run_id" --exit-status
test "$(gh run view "$run_id" --json headSha --jq .headSha)" = "$release_sha"
test "$(gh run view "$run_id" --json conclusion --jq .conclusion)" = success

git tag -a v0.4.0 "$release_sha" -m "strided-rs v0.4.0"
git push origin v0.4.0
git switch --detach v0.4.0
test -z "$(git status --porcelain)"
```

Create each package archive from the tag and verify Cargo recorded the tagged
commit in `.cargo_vcs_info.json`:

```bash
version=0.4.0
expected=$(git rev-parse HEAD)
for crate in \
  strided-traits strided-view strided-perm strided-kernel \
  strided-einsum2 strided-opteinsum mdarray-opteinsum \
  ndarray-opteinsum strided-rs
do
  cargo package -p "$crate" --no-verify
  archive="target/package/${crate}-${version}.crate"
  vcs_info=$(tar -xOf "$archive" "${crate}-${version}/.cargo_vcs_info.json")
  actual=$(printf '%s' "$vcs_info" | python3 -c 'import json, sys; print(json.load(sys.stdin)["git"]["sha1"])')
  dirty=$(printf '%s' "$vcs_info" | python3 -c 'import json, sys; print(str(json.load(sys.stdin)["git"].get("dirty", False)).lower())')
  test "$actual" = "$expected"
  test "$dirty" = false
done
```

Do not publish if an archive names another commit or reports a dirty worktree.

## 3. Publish in dependency order

Publish one layer at a time in this order:

1. `strided-traits`
2. `strided-view`
3. `strided-perm`
4. `strided-kernel`
5. `strided-einsum2`
6. `strided-opteinsum`
7. `mdarray-opteinsum` and `ndarray-opteinsum` (either order)
8. `strided-rs`

For each crate, publish from the unchanged tag checkout:

```bash
cargo publish -p <crate>
```

After each publish, wait until crates.io serves that exact version before
publishing a dependent crate:

```bash
cargo info <crate>@0.4.0
```

Registry indexing is asynchronous; retry `cargo info` rather than changing a
manifest, repackaging with different metadata, or publishing dependents early.
After the facade is visible, leave the detached checkout unchanged and verify
`git status --porcelain` is empty.
