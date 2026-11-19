#!/usr/bin/env bash
# Build, run the bit-exact tests, then print the GEMM ladder table.
#   scripts/gemm_check.sh                 # default shapes
#   scripts/gemm_check.sh 1024 1024 1024  # one shape
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build >/dev/null
cmake --build build -j 2>&1 | grep -E "error|warning: (implicit|unused)|Error" || true
./build/test_kernels | tail -3
if command -v taskset >/dev/null && taskset -c "${CPU:-2}" true 2>/dev/null; then
  taskset -c "${CPU:-2}" ./build/bench_gemm "$@"
else ./build/bench_gemm "$@"; fi
