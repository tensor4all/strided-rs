#!/usr/bin/env bash
# File, update or close the tracking issue for the weekly `Packaged crate tests`
# run on main (#517). Called by the `report` job of package-tests.yml.
#
# Inputs (environment):
#   GITHUB_REPOSITORY, SHA, RUN_URL      where and what was checked
#   LIST_RESULT, IDENTITY_RESULT, PACKAGE_RESULT
#                                        needs.<job>.result of the three jobs
#   IDENTITY_REPORT                      published_identity.py output, verbatim
#   GH (optional)                        gh executable; tests point it at a stub
#
# A failure, or a job that did not finish (cancelled / skipped without its
# upstream failing), files one issue with label `package-tests-failure`, or
# comments on the open one. A fully green run comments on and closes it, so an
# open issue always means "still red". If the issue cannot be filed, this
# script fails, which keeps the run red instead of losing the signal.
set -euo pipefail

GH="${GH:-gh}"
LABEL=package-tests-failure
MARK="<!-- package-tests-failure -->"

bad=()
for pair in "list:${LIST_RESULT}" "identity:${IDENTITY_RESULT}" "package:${PACKAGE_RESULT}"; do
  job="${pair%%:*}" result="${pair#*:}"
  case "$result" in
    success) ;;
    # package is skipped only because list did not succeed; list is then in `bad`.
    skipped) [ "$job" = package ] || bad+=("${job}=${result}") ;;
    *) bad+=("${job}=${result}") ;;
  esac
done

"$GH" api -X POST "repos/${GITHUB_REPOSITORY}/labels" \
  -f name="$LABEL" -f color=B60205 \
  -f description="Packaged crate tests failed on main (filed by the workflow)" >/dev/null 2>&1 || true

existing=$("$GH" issue list -R "$GITHUB_REPOSITORY" --state open --label "$LABEL" \
  --json number,body --limit 100 \
  --jq "[.[] | select(.body | contains(\"${MARK}\"))] | .[0].number // empty")

body=$(mktemp)
trap 'rm -f "$body"' EXIT

if [ "${#bad[@]}" -eq 0 ]; then
  if [ -n "$existing" ]; then
    printf 'Packaged crate tests are green again at `%s` (%s). Closing.\n' "$SHA" "$RUN_URL" > "$body"
    "$GH" issue comment "$existing" -R "$GITHUB_REPOSITORY" --body-file "$body"
    "$GH" issue close "$existing" -R "$GITHUB_REPOSITORY"
    echo "closed #${existing}"
  else
    echo "all jobs green; nothing to file"
  fi
  exit 0
fi

{
  echo "$MARK"
  echo "The weekly \`Packaged crate tests\` run failed on main."
  echo
  echo "- commit: \`${SHA}\`"
  echo "- run: ${RUN_URL}"
  echo "- jobs not green: \`${bad[*]}\`"
  echo
  echo "\`published content identity\` output, verbatim (\`exit 0\` ok, \`1\` mismatch, \`2\` could not check):"
  echo
  echo "~~~"
  if [ -n "${IDENTITY_REPORT:-}" ]; then printf '%s\n' "$IDENTITY_REPORT"; else echo "(no output: the job did not reach the comparison)"; fi
  echo "~~~"
  echo
  echo "Reading it: the comparison is **this commit of main** against the .crate on crates.io for the"
  echo "version main's manifest declares, not against a release tag. \`MISMATCH\` here while the same"
  echo "check is clean at the release tag means main changed shipped files after publishing without"
  echo "moving the version (the 0.14.1 / #452 case); the remedy is a version bump on main, not a tag move."
  echo "A \`cargo test in the .crate\` failure that cannot resolve a dependency item usually has the same"
  echo "cause: the packaged crate builds against the published, older dependency."
  echo
  echo "Reproduce locally:"
  echo
  echo "~~~"
  echo "python3 scripts/published_identity.py . --ref ${SHA}"
  echo "python3 scripts/published_identity.py . --ref <crate>-v<version>   # the release tag"
  echo "~~~"
} > "$body"

if [ -n "$existing" ]; then
  "$GH" issue comment "$existing" -R "$GITHUB_REPOSITORY" --body-file "$body"
  echo "commented on #${existing}"
else
  "$GH" issue create -R "$GITHUB_REPOSITORY" \
    --title "Packaged crate tests fail on main" \
    --label "$LABEL" \
    --body-file "$body"
fi
