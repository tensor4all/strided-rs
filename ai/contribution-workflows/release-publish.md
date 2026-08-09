# Publishing a strided-rs release

Publish only a commit that has already landed on `main`. The release tag must
identify that exact commit, and every crate must be published from a clean
checkout of the tag.

## 1. Prepare and validate `main`

Complete all version and dependency changes in a reviewed commit before the
release. Do not edit manifests while publishing: crates.io must receive the
same manifests recorded by the tag.

```bash
set -euo pipefail

git switch main
git pull --ff-only origin main
cargo fmt --all -- --check
cargo test --workspace
```

Confirm all nine package file lists contain their license and provenance files:

```bash
set -euo pipefail

for crate in \
  strided-traits strided-view strided-perm strided-kernel \
  strided-einsum2 strided-opteinsum mdarray-opteinsum \
  ndarray-opteinsum strided-rs
do
  package_files=$(cargo package -p "$crate" --list)
  printf '%s\n' "$package_files"
  for required in LICENSE-APACHE LICENSE-MIT NOTICE; do
    grep -Fxq "$required" <<<"$package_files"
  done
  case "$crate" in
    strided-traits|strided-view|strided-perm|strided-kernel)
      grep -Fxq THIRD-PARTY-LICENSES <<<"$package_files"
      ;;
    *)
      if grep -Fxq THIRD-PARTY-LICENSES <<<"$package_files"; then
        echo "unexpected THIRD-PARTY-LICENSES in $crate" >&2
        exit 1
      fi
      ;;
  esac
done
```

Every package must contain `LICENSE-APACHE`, `LICENSE-MIT`, and its
package-specific `NOTICE`. Only `strided-traits`, `strided-view`,
`strided-kernel`, and `strided-perm` contain ported or license-derived code, so
only those archives contain `THIRD-PARTY-LICENSES`.

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

The detached tag checkout is the publication source. Do not create all nine
archives at once: Cargo cannot package a crate whose v0.4 workspace
prerequisites are not yet available from crates.io.

## 3. Package, inspect, and publish in dependency order

Process one crate completely before starting the next, in this exact order:

1. `strided-traits`
2. `strided-view`
3. `strided-perm`
4. `strided-kernel`
5. `strided-einsum2`
6. `strided-opteinsum`
7. `mdarray-opteinsum`
8. `ndarray-opteinsum`
9. `strided-rs`

The adapters occupy the same dependency layer and may be processed in either
order. For every crate, first query crates.io for the exact version. An absent
version is packaged, inspected, dry-run, published, and awaited. An existing
version is skipped only after its registry archive passes the same provenance
checks. Thus every prerequisite is registry-visible before Cargo packages a
dependent crate, and the first new archive is fully inspected before the first
irreversible publish.

```bash
set -euo pipefail

version=0.4.0
expected=$(git rev-parse HEAD)
test "$(git describe --exact-match --tags HEAD)" = "v$version"

download_dir=$(mktemp -d)
trap 'rm -rf "$download_dir"' EXIT

verify_archive() {
  local crate=$1 archive=$2 prefix="${1}-${version}"
  local vcs_info actual dirty archive_files

  vcs_info=$(tar -xOf "$archive" "$prefix/.cargo_vcs_info.json")
  actual=$(printf '%s' "$vcs_info" | python3 -c 'import json, sys; print(json.load(sys.stdin)["git"]["sha1"])')
  dirty=$(printf '%s' "$vcs_info" | python3 -c 'import json, sys; print(str(json.load(sys.stdin)["git"].get("dirty", False)).lower())')
  test "$actual" = "$expected"
  test "$dirty" = false

  for required in LICENSE-APACHE LICENSE-MIT NOTICE; do
    tar -xOf "$archive" "$prefix/$required" | cmp - "$crate/$required"
  done
  case "$crate" in
    strided-traits|strided-view|strided-perm|strided-kernel)
      tar -xOf "$archive" "$prefix/THIRD-PARTY-LICENSES" |
        cmp - "$crate/THIRD-PARTY-LICENSES"
      ;;
    *)
      archive_files=$(tar -tf "$archive")
      if grep -q '/THIRD-PARTY-LICENSES$' <<<"$archive_files"; then
        echo "unexpected THIRD-PARTY-LICENSES in $crate" >&2
        exit 1
      fi
      ;;
  esac
}

for crate in \
  strided-traits strided-view strided-perm strided-kernel \
  strided-einsum2 strided-opteinsum mdarray-opteinsum \
  ndarray-opteinsum strided-rs
do
  test -z "$(git status --porcelain)"

  registry_archive="$download_dir/${crate}-${version}.crate"
  if ! http_status=$(curl --location --silent --show-error \
    --user-agent "strided-rs-release/$version" \
    --output "$registry_archive" --write-out '%{http_code}' \
    "https://crates.io/api/v1/crates/$crate/$version/download"); then
    echo "failed to query crates.io for $crate@$version" >&2
    exit 1
  fi

  case "$http_status" in
    200)
      verify_archive "$crate" "$registry_archive"
      echo "$crate@$version already exists and matches tag; skipping"
      continue
      ;;
    404)
      rm -f "$registry_archive"
      ;;
    *)
      echo "unexpected HTTP status $http_status for $crate@$version" >&2
      exit 1
      ;;
  esac

  cargo package -p "$crate" --no-verify
  archive="target/package/${crate}-${version}.crate"
  verify_archive "$crate" "$archive"

  cargo publish -p "$crate" --dry-run
  cargo publish -p "$crate"

  visible=false
  for attempt in {1..30}; do
    if cargo info "$crate@$version"; then
      visible=true
      break
    fi
    sleep 10
  done
  test "$visible" = true
done

test -z "$(git status --porcelain)"
```

The visibility wait is bounded to 30 attempts and retries because registry
indexing is asynchronous. If it expires, rerun the same unchanged block: this
is safe only because every existing version is downloaded and
provenance-verified against the tag before it is skipped. Any network failure,
HTTP result other than 200 or 404, or archive mismatch aborts the release. Do
not edit a manifest, repackage with different metadata, or publish a dependent
early. After the facade is visible, leave the detached checkout unchanged.
