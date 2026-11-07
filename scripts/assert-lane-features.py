#!/usr/bin/env python3
"""Assert the feature set cargo actually compiled, from its JSON messages.

Reads a `cargo build` or `cargo test --no-run` `--message-format=json` stream on
stdin. `WANT_BASIC` is the exact feature set every `strided-basic` unit must
report (space separated, empty for none) and `WANT_PULP` is `yes` or `no`. Exits
non-zero with an `::error::` line unless every `strided-basic` unit matches and
`pulp` is built exactly when expected, so a feature lane cannot pass while
compiling something other than its name (tensor4all/strided-rs#215).

A lane asserts two graphs, because they are not the same graph: the library
build and the test build (`cargo test --no-run` adds dev-dependencies, whose
feature requirements take part in the unification).
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    want_basic = set(os.environ.get("WANT_BASIC", "").split())
    want_pulp = os.environ.get("WANT_PULP", "no") == "yes"

    units: list[set[str]] = []
    pulp = False
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("reason") != "compiler-artifact":
            continue
        target = message["target"]["name"]
        if target == "strided_basic":
            units.append(set(message["features"]))
        elif target == "pulp":
            pulp = True

    print(
        f"strided-basic units: {[sorted(unit) for unit in units]}; "
        f"pulp built: {pulp}; expected {sorted(want_basic)} with pulp={want_pulp}"
    )
    if not units:
        print(
            "::error::no strided-basic unit in the cargo output; the build did not run",
            file=sys.stderr,
        )
        return 1
    if any(unit != want_basic for unit in units) or pulp != want_pulp:
        print(
            f"::error::built strided-basic {[sorted(unit) for unit in units]} "
            f"with pulp={pulp}; expected {sorted(want_basic)} with pulp={want_pulp}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
