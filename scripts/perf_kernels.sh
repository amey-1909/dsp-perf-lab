#!/usr/bin/env bash
# IPC / instruction table for FIR, int8 GEMM and SAD, one variant at a time.
#
#   scripts/perf_kernels.sh                 # all kernels, all variants
#   scripts/perf_kernels.sh fir sad         # chosen kernels (fir gemm sad sadx4)
#   CPU=3 scripts/perf_kernels.sh fir       # pin to another core
#
# Needs working hardware counters:  sudo sysctl kernel.perf_event_paranoid=1
# For each (kernel, variant) it first runs a short calibration (no perf) to
# pick an iteration count of about one second, then runs again under
# `perf stat` and normalises by the work done (output sample / MAC / SAD).
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/kernel_run
[[ -x "$BIN" ]] || { echo "build first: cmake -S . -B build && cmake --build build -j" >&2; exit 1; }
command -v perf >/dev/null || { echo "perf not found (sudo apt install linux-tools-common linux-tools-generic linux-tools-\$(uname -r))" >&2; exit 1; }

CPU="${CPU:-2}"
PIN=()
if command -v taskset >/dev/null && taskset -c "$CPU" true 2>/dev/null; then PIN=(taskset -c "$CPU"); fi

kernels=("$@")
[[ ${#kernels[@]} -gt 0 ]] || kernels=(fir gemm sad sadx4)
variants=(ref autovec avx2)

if ! ${PIN[@]+"${PIN[@]}"} perf stat -x, -e cycles,instructions true >/dev/null 2>&1; then
  echo "perf cannot read cycles/instructions here. Try: sudo sysctl kernel.perf_event_paranoid=1" >&2
  echo "(inside a VM or WSL the hardware counters may not be exposed at all)" >&2
  exit 1
fi

field() { sed -n "s/.*[ ]$1=\([^ ]*\).*/\1/p" <<<"$2"; }

printf "%-6s %-8s %-8s %10s %12s %12s %7s\n" kernel variant unit "ns/unit" "instr/unit" "cycles/unit" IPC
for k in "${kernels[@]}"; do
  for v in "${variants[@]}"; do
    cal=$(${PIN[@]+"${PIN[@]}"} "$BIN" "$k" "$v" 20 | grep '^RESULT') || { echo "$k $v: refused or failed" >&2; continue; }
    ns_iter=$(field ns_per_iter "$cal")
    iters=$(awk -v ns="$ns_iter" 'BEGIN { n = int(1e9 / ns); if (n < 50) n = 50; print n }')

    out=$(mktemp); err=$(mktemp)
    ${PIN[@]+"${PIN[@]}"} perf stat -x, -e cycles,instructions "$BIN" "$k" "$v" "$iters" >"$out" 2>"$err"
    res=$(grep '^RESULT' "$out")
    upi=$(field units_per_iter "$res"); unit=$(field unit "$res"); ns_unit=$(field ns_per_unit "$res")
    cyc=$(awk -F, '$3 ~ /^cycles/ {print $1; exit}' "$err")
    ins=$(awk -F, '$3 ~ /^instructions/ {print $1; exit}' "$err")
    rm -f "$out" "$err"
    awk -v k="$k" -v v="$v" -v u="$unit" -v ns="$ns_unit" -v c="$cyc" -v i="$ins" -v n="$iters" -v w="$upi" 'BEGIN {
      work = n * w
      printf "%-6s %-8s %-8s %10.4f %12.3f %12.3f %7.2f\n", k, v, u, ns, i / work, c / work, i / c }'
  done
done
echo
echo "instr/unit = instructions per output sample / MAC / SAD. Fewer is better. IPC alone is NOT the goal:"
echo "a SIMD kernel can have LOWER IPC than scalar and still be much faster, because each instruction does more work."
