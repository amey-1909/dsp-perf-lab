#!/usr/bin/env bash
# Static pipeline analysis of the AVX2 GEMM inner loop with llvm-mca:
# predicted IPC, block reciprocal throughput, and execution-port pressure.
# Usage: scripts/mca.sh [cpu]    e.g. haswell, skylake, znver3, icelake-server
set -euo pipefail
cd "$(dirname "$0")/.."
CPU="${1:-skylake}"
gcc -O3 -mavx2 -mfma -DMCA_MARKERS -Iinclude -S kernels/avx2/gemm_s8_avx2.c -o - \
  | llvm-mca -mcpu="$CPU" -iterations=200 --bottleneck-analysis --timeline --timeline-max-iterations=3
