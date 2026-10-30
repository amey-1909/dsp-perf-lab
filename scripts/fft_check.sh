#!/usr/bin/env bash
# Build, run the full correctness suite (every FFT step vs a naive DFT), then
# print the step-ladder table. Extra arguments are FFT sizes for bench_fft.
#   scripts/fft_check.sh            # default sizes
#   scripts/fft_check.sh 1024 8192
set -euo pipefail
cd "$(dirname "$0")/.."

cmake -S . -B build >/dev/null
cmake --build build -j
./build/test_kernels

CPU="${CPU:-2}"
if command -v taskset >/dev/null 2>&1 && taskset -c "$CPU" true 2>/dev/null; then
  exec taskset -c "$CPU" ./build/bench_fft "$@"
fi
exec ./build/bench_fft "$@"
