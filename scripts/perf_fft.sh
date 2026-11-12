#!/usr/bin/env bash
# Hardware-counter table for every FFT step: time, instructions, cycles, IPC and
# L1 misses per FFT, for each size. Output is a Markdown table for the README.
#
#   scripts/perf_fft.sh                    # sizes 256 1024 4096 16384 65536
#   scripts/perf_fft.sh 1024 65536         # chosen sizes
#   CPU=3 scripts/perf_fft.sh 1024         # pin to another core
#
# Needs working counters:  sudo sysctl kernel.perf_event_paranoid=1
#
# Counters of one run include the start-up work of fft_run (self-check against
# the reference, warm-up, allocation). To remove it, each (step, n) is run twice,
# with K and 2K iterations, and the per-FFT value is (counts(2K) - counts(K)) / K.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=./build/fft_run
[[ -x "$BIN" ]] || { echo "build first: cmake -S . -B build && cmake --build build -j" >&2; exit 1; }
command -v perf >/dev/null || { echo "perf not found (sudo apt install linux-tools-common linux-tools-generic linux-tools-\$(uname -r))" >&2; exit 1; }

CPU="${CPU:-2}"
PIN=()
if command -v taskset >/dev/null && taskset -c "$CPU" true 2>/dev/null; then PIN=(taskset -c "$CPU"); fi

sizes=("$@")
[[ ${#sizes[@]} -gt 0 ]] || sizes=(256 1024 4096 16384 65536)

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

# one perf run -> "cycles instructions l1miss ns_per_fft" (l1miss = -1 if unsupported)
run_once() {
  local which=$1 n=$2 iters=$3 out
  out=$(${PIN[@]+"${PIN[@]}"} perf stat -x, -e cycles,instructions,L1-dcache-load-misses -o "$TMP" -- \
        "$BIN" "$which" "$n" "$iters") || { echo "FAIL"; return; }
  local ns cyc ins l1
  ns=$(sed -n 's/.* \([0-9.]*\) ns\/FFT.*/\1/p' <<<"$out")
  cyc=$(awk -F, 'index($3,"cycles") && !index($3,"L1") {print $1}' "$TMP" | head -1)
  ins=$(awk -F, 'index($3,"instructions") {print $1}' "$TMP" | head -1)
  l1=$(awk -F, 'index($3,"L1-dcache") {print $1}' "$TMP" | head -1)
  [[ "$cyc" =~ ^[0-9]+$ && "$ins" =~ ^[0-9]+$ ]] || { echo "NOCOUNTERS"; return; }
  [[ "$l1" =~ ^[0-9]+$ ]] || l1=-1
  echo "$cyc $ins $l1 $ns"
}

# steps: the reference, then every fft_s<N>_*.c in order
names=(ref)
for f in $(ls kernels/avx2/fft/fft_s[0-9]*_*.c | sort -V); do
  b=$(basename "$f" .c); b=${b#fft_}
  names+=("$b")
done

echo "| n | step | ns/FFT | instr/FFT | cycles/FFT | IPC | L1 miss/FFT |"
echo "|--:|------|-------:|----------:|-----------:|----:|------------:|"
for n in "${sizes[@]}"; do
  # about 0.4 s of work for the fast steps; the minimum keeps big sizes meaningful
  k=$(awk -v n="$n" 'BEGIN { k = int(100000 * 1024 / n); if (k < 400) k = 400; print k }')
  for nm in "${names[@]}"; do
    which=$nm; [[ $nm == ref ]] || which=${nm%%_*}      # "s4_direct8" -> "s4"
    a=$(run_once "$which" "$n" "$k")
    b=$(run_once "$which" "$n" "$((2 * k))")
    if [[ $a == FAIL || $b == FAIL ]]; then echo "| $n | $nm | (refused or failed) | | | | |"; continue; fi
    if [[ $a == NOCOUNTERS || $b == NOCOUNTERS ]]; then
      echo "perf cannot read cycles/instructions. Try: sudo sysctl kernel.perf_event_paranoid=1 (VMs may not expose counters)" >&2
      exit 1
    fi
    read -r c1 i1 m1 t1 <<<"$a"
    read -r c2 i2 m2 t2 <<<"$b"
    awk -v n="$n" -v nm="$nm" -v k="$k" -v c1="$c1" -v i1="$i1" -v m1="$m1" -v c2="$c2" -v i2="$i2" -v m2="$m2" -v t="$t2" 'BEGIN {
      cyc = (c2 - c1) / k; ins = (i2 - i1) / k;
      ipc = cyc > 0 ? ins / cyc : 0;
      l1 = (m1 >= 0 && m2 >= 0) ? sprintf("%.0f", (m2 - m1) / k) : "n/a";
      printf "| %d | %s | %.0f | %.0f | %.0f | %.2f | %s |\n", n, nm, t, ins, cyc, ipc, l1 }'
  done
done
echo
echo "Per-FFT figures = (counts(2K iters) - counts(K iters)) / K, so start-up work cancels out."
echo "ns/FFT includes one memcpy of the input, same as bench_fft."
