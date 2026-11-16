#!/bin/bash
# The observer (this script + the shell) runs in the other L3 domain; the measured
# cells are pinned to one L3 domain: index3 cache 0 is cores 0-7 and 16-23.
# Core 0 is skipped on purpose — foreign processes ("fbench verify --threads 1" and
# "tlbench-before") were pinned there and a first version of this campaign
# measured them instead of the host (see docs/worklogs, §3).
set -u
cd "$(dirname "$0")"
OUT=raw
mkdir -p "$OUT"
BIN=target/release/mace-ab
log="$OUT/campaign.txt"
: > "$log"
ONE=4          # 1T cell
FOUR=4,5,6,7   # 4T cell: physical cores 4-7, same index3 cache
OBSERVER=8-15,24-31

# Drift guard: the same anchor cell at both ends of the campaign has to agree
# within 15 %, otherwise the run measured a changing machine and is worthless.
anchor() {
  taskset -c "$ONE" "$BIN" --cell cg-fwd --dtype f64 --threads 1 --atoms 128 \
    --samples 15 --target-ms 20 2>/dev/null
}

echo "=== anchor start $(date -Is) cpus=$ONE" >> "$log"
start=$(anchor)
echo "$start" >> "$log"
if [ -z "$start" ]; then echo "=== FAILED anchor start" >> "$log"; exit 1; fi

for atoms in 128 1024; do
  for dtype in f64 f32; do
    for threads in 1 4; do
      for cell in cg-fwd cg-vjp; do
        cpus="$ONE"
        [ "$threads" = 4 ] && cpus="$FOUR"
        echo "=== $cell $dtype ${threads}T atoms=$atoms cpus=$cpus $(date -Is)" >> "$log"
        taskset -c "$cpus" "$BIN" --cell "$cell" --dtype "$dtype" \
          --threads "$threads" --atoms "$atoms" --samples 15 --raw-dir "$OUT" \
          >> "$log" 2>&1 || echo "FAILED $cell $dtype $threads $atoms" >> "$log"
      done
    done
  done
done

echo "=== anchor end $(date -Is) cpus=$ONE" >> "$log"
finish=$(anchor)
echo "$finish" >> "$log"
if [ -z "$finish" ]; then echo "=== FAILED anchor end" >> "$log"; exit 1; fi

python3 - "$start" "$finish" >> "$log" 2>&1 <<'PY'
import json, sys
start, end = json.loads(sys.argv[1]), json.loads(sys.argv[2])
failed = False
for side, read in (("strided", lambda r: r["timings"]["strided"]["median_ns"]),
                   ("llmlip", lambda r: r["timings"]["llmlip"]["median_ns"])):
    a, b = read(start), read(end)
    drift = abs(b - a) / max(a, b)
    print(f"drift {side}: start {a / 1e6:.1f} ms, end {b / 1e6:.1f} ms, {100 * drift:.1f} %")
    if drift > 0.15:
        print(f"DRIFT {side} exceeded 15 %, discard this campaign")
        failed = True
sys.exit(1 if failed else 0)
PY
if [ $? -eq 0 ]; then
  echo "=== done $(date -Is)" >> "$log"
else
  echo "=== FAILED drift $(date -Is)" >> "$log"
  exit 1
fi
