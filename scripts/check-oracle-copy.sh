#!/usr/bin/env bash
# `strided-perm/src/block/tests/oracle.rs` is a copy of the strided-basic one.
# The alternative — `#[path = "../../../strided-basic/..."]` — escapes the
# package boundary, so the file is absent from the published `.crate` and
# `cargo test` fails there (#397 did exactly that).
set -euo pipefail
diff -u strided-basic/src/block/tests/oracle.rs strided-perm/src/block/tests/oracle.rs
echo "oracle copies are identical"
