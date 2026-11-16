#!/bin/bash
# Observer (this script + shell) lives on the other L3 domain; measured cells are
# pinned to cores 0-3, which share index3 cache 0 (0-7,16-23).
set -u
cd "$(dirname "$0")"
OUT=raw
mkdir -p "$OUT"
BIN=target/release/mace-ab
log="$OUT/campaign.txt"
: > "$log"
for atoms in 128 1024; do
  for dtype in f64 f32; do
    for threads in 1 4; do
      for cell in cg-fwd cg-vjp; do
        cpus="0"
        [ "$threads" = 4 ] && cpus="0,1,2,3"
        echo "=== $cell $dtype ${threads}T atoms=$atoms cpus=$cpus $(date -Is)" >> "$log"
        taskset -c "$cpus" "$BIN" --cell "$cell" --dtype "$dtype" \
          --threads "$threads" --atoms "$atoms" --samples 15 --raw-dir "$OUT" \
          >> "$log" 2>&1 || echo "FAILED $cell $dtype $threads $atoms" >> "$log"
      done
    done
  done
done
echo "=== done $(date -Is)" >> "$log"
