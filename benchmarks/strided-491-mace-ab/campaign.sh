#!/bin/bash
# The observer (this script + the shell) runs in the other L3 domain; the measured
# cells are pinned to one L3 domain: index3 cache 0 is cores 0-7 and 16-23.
# Core 0 is skipped on purpose — foreign processes ("fbench verify --threads 1" and
# "tlbench-before") were pinned there and a first version of this campaign
# measured them instead of the host (see docs/worklogs, §3).
set -u
cd "$(dirname "$0")"
OUT=${OUT:-raw}
mkdir -p "$OUT"
BIN=${BIN:-target/release/mace-ab}
log="$OUT/campaign.txt"
: > "$log"
ONE=4          # 1T cell
FOUR=4,5,6,7   # 4T cell: physical cores 4-7, same index3 cache
OBSERVER=8-15,24-31

# CPU affinity is a Linux facility. macOS has no per-thread core binding, so on
# a host without `taskset` the same cells run unpinned and the log says so; the
# thread counts still fix the work each side is allowed to open. Set PIN=1 to
# require pins (a missing `taskset` then fails), PIN=0 to run unpinned anywhere.
PIN=${PIN:-auto}
if [ "$PIN" = auto ]; then
  if command -v taskset >/dev/null 2>&1; then PIN=1; else PIN=0; fi
fi
run_pinned() {
  cpus=$1
  shift
  if [ "$PIN" = 1 ]; then
    taskset -c "$cpus" "$@"
  else
    "$@"
  fi
}
# BSD `date` (macOS) has no -I, and the transcript is read by humans only.
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
# A cell that allocates a gigabyte (the 1024-atom VJP) leaves the OS reclaiming
# memory after the process exits. The three no-pause campaigns on the M5 ended
# with the 1T f64 anchor 19.8-26.3 % slow (13.7-18.0 ms against 11.0-13.5 ms at
# the start); with SETTLE=5 the accepted pair's anchors moved 1.3 % and
# 1.7-2.0 %. Linux did not show it, so the delay defaults to 0 and the M5 runs
# set SETTLE=5; it also gives the fan a moment between cells. A zero keeps the
# evo2 transcript identical to #493/#498.
SETTLE=${SETTLE:-0}
settle() { if [ "$SETTLE" -gt 0 ]; then sleep "$SETTLE"; fi; }

# Drift guard: the same anchor cell at both ends of the campaign has to agree
# within 15 %, otherwise the run measured a changing machine and is worthless.
anchor() {
  settle
  run_pinned "$ONE" "$BIN" --cell cg-fwd --dtype f64 --threads 1 --atoms 128 \
    --samples 15 --target-ms 20 2>/dev/null
}

echo "=== anchor start $(now) cpus=$ONE pin=$PIN settle=$SETTLE" >> "$log"
start=$(anchor)
echo "$start" >> "$log"
if [ -z "$start" ]; then echo "=== FAILED anchor start" >> "$log"; exit 1; fi

failed=0
for atoms in 128 1024; do
  for dtype in f64 f32; do
    for threads in 1 4; do
      for cell in cg-fwd cg-vjp; do
        cpus="$ONE"
        [ "$threads" = 4 ] && cpus="$FOUR"
        echo "=== $cell $dtype ${threads}T atoms=$atoms cpus=$cpus $(now)" >> "$log"
        settle
        if ! run_pinned "$cpus" "$BIN" --cell "$cell" --dtype "$dtype" \
          --threads "$threads" --atoms "$atoms" --samples 15 --raw-dir "$OUT" \
          >> "$log" 2>&1; then
          echo "FAILED $cell $dtype $threads $atoms" >> "$log"
          failed=1
        fi
      done
    done
  done
done

echo "=== anchor end $(now) cpus=$ONE pin=$PIN settle=$SETTLE" >> "$log"
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
drift_ok=$?
# A record that is missing or a cell that died makes the run worthless even when
# the anchors agree, so the exit status has to carry both.
if [ $drift_ok -ne 0 ]; then
  echo "=== FAILED drift $(now)" >> "$log"
  exit 1
fi
if [ $failed -ne 0 ]; then
  echo "=== FAILED cells $(now)" >> "$log"
  exit 1
fi
echo "=== done $(now)" >> "$log"
