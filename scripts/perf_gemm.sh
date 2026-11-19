#!/usr/bin/env bash
# Per-step counters for one shape: ns/GEMM, instr, cycles, IPC, and optionally port 0/1/5 uops.
#   scripts/perf_gemm.sh                  # 128 128 256
#   scripts/perf_gemm.sh 256 256 256
#   PORTS=1 scripts/perf_gemm.sh          # add port utilization (Intel Skylake-family events)
# Start-up cancels: each step is run with K and 2K iterations and subtracted.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=./build/gemm_run
[[ -x $BIN ]] || { echo "build first" >&2; exit 1; }
M=${1:-128}; N=${2:-128}; K=${3:-256}; CPU=${CPU:-2}
PIN=(); command -v taskset >/dev/null && taskset -c "$CPU" true 2>/dev/null && PIN=(taskset -c "$CPU")
EV=cycles,instructions
[[ ${PORTS:-0} == 1 ]] && EV=$EV,uops_dispatched_port.port_0,uops_dispatched_port.port_1,uops_dispatched_port.port_5
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
iters=$(awk -v m=$M -v n=$N -v k=$K 'BEGIN{i=int(2e9/(m*n*k)); if(i<200)i=200; print i}')
get() { ${PIN[@]+"${PIN[@]}"} perf stat -x, -e "$EV" -o "$TMP" -- "$BIN" "$1" $M $N $K "$2" >/dev/null || return 1
        awk -F, '$1 ~ /^[0-9]+$/ {printf "%s ", $1}' "$TMP"; }
names=(ref autovec); for f in $(ls kernels/avx2/gemm/gemm_s[0-9]*_*.c | sort -V); do b=$(basename "$f" .c); names+=("${b#gemm_}"); done
echo "| step | instr/GEMM | cycles/GEMM | IPC | cycles/MAC | p0 | p1 | p5 |"
echo "|------|-----------:|------------:|----:|-----------:|---:|---:|---:|"
for nm in "${names[@]}"; do
  w=$nm; [[ $nm == ref || $nm == autovec ]] || w=${nm%%_*}
  a=$(get "$w" "$iters") || { echo "| $nm | (failed) |||||||"; continue; }
  b=$(get "$w" "$((2*iters))") || continue
  awk -v nm="$nm" -v it="$iters" -v macs="$((M*N*K))" -v a="$a" -v b="$b" 'BEGIN{
    split(a,x," "); split(b,y," ");
    cyc=(y[1]-x[1])/it; ins=(y[2]-x[2])/it;
    p0=y[3]!=""?sprintf("%.0f%%",100*(y[3]-x[3])/(y[1]-x[1])):"-";
    p1=y[4]!=""?sprintf("%.0f%%",100*(y[4]-x[4])/(y[1]-x[1])):"-";
    p5=y[5]!=""?sprintf("%.0f%%",100*(y[5]-x[5])/(y[1]-x[1])):"-";
    printf "| %s | %.0f | %.0f | %.2f | %.3f | %s | %s | %s |\n",nm,ins,cyc,ins/cyc,cyc/macs,p0,p1,p5}'
done
