#!/bin/bash
# usage: 522-disasm.sh <path to mace-ab release binary>
# Counting rule: every instruction line objdump -d prints for the symbol's [addr, addr+size) range, nops included ("insns"),
# plus the count of lines naming a zmm register, containing vcvt, or vfmadd.
B=${1:?binary}
set -eu
cnt0() { grep -c "$@" || true; }
# A symbol that is missing or ambiguous is an error, never an "identical" line.
sym() { # $1 = grep pattern -> "addr size"
  m=$(nm -C -S "$B" | grep -- "$1" || true)
  [ "$(echo "$m" | cnt0 .)" = 1 ] || { echo "symbol lookup failed or ambiguous ($(echo "$m" | cnt0 .) matches): $1" >&2; exit 1; }
  echo "$m" | cut -d" " -f1,2
}
cnt() { # $1 start $2 size -> insn lines
  st=$((16#$1)); en=$((st+16#$2))
  [ "$en" -gt "$st" ] || { echo "empty range" >&2; exit 1; }
  objdump -d --no-show-raw-insn --start-address=$st --stop-address=$en "$B" | grep -E '^\s+[0-9a-f]+:' | sed -E 's/^\s+[0-9a-f]+:\s+//'
}
echo "== V4 lane kernels RowF32<A,B> vs RowF64<A,B>"
for k in "2 1" "2 2" "2 3" "3 1" "3 2" "3 3" "3 4" "3 5" "3 6" "3 7"; do
  set -- $k; out=""
  for t in 32 64; do
    l=$(sym "V4 as pulp::Simd>::vectorize::Impl<strided_multilinear::simd::RowF$t<$1, $2>>>")
    t_=$(cnt $l)
    out="$out f$t: insns=$(echo "$t_" | wc -l) nop=$(echo "$t_" | cnt0 nop) zmm=$(echo "$t_" | cnt0 zmm) vcvt=$(echo "$t_" | cnt0 vcvt) fma=$(echo "$t_" | cnt0 vfmadd) mul=$(echo "$t_" | cnt0 -E 'vmulp[sd]') add=$(echo "$t_" | cnt0 -E 'vaddp[sd]') mov=$(echo "$t_" | cnt0 -E 'vmovup[sd]');"
  done
  echo "Row<$1,$2>:$out"
done
echo "== SegPolyPlan<f32> vs <f64>"
for fn in execute run_task accumulate_output compile_with_block_rows; do
  out=""
  for t in f32 f64; do
    l=$(sym "SegPolyPlan<$t>>::$fn\$")
    t_=$(cnt $l)
    out="$out $t: insns=$(echo "$t_" | wc -l) vcvt=$(echo "$t_" | cnt0 vcvt) cvt=$(echo "$t_" | cnt0 -E '\bcvt');"
  done
  echo "$fn:$out"
done
