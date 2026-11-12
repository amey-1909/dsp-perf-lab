#!/usr/bin/env bash
# Build and run the comparison with FFTW (single precision).
#   scripts/bench_fftw.sh              # sizes 256 1024 4096 16384 65536
#   scripts/bench_fftw.sh 1024 8192    # chosen sizes
#   CPU=3 scripts/bench_fftw.sh        # pin to another core
# Needs: sudo apt install libfftw3-dev
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build >/dev/null
cmake --build build -j >/dev/null
if [[ ! -x build/bench_fftw ]]; then
  echo "bench_fftw was not built: FFTW (single precision) not found." >&2
  echo "Install it with:  sudo apt install libfftw3-dev   then run this script again." >&2
  exit 1
fi
CPU="${CPU:-2}"
if command -v taskset >/dev/null 2>&1 && taskset -c "$CPU" true 2>/dev/null; then
  exec taskset -c "$CPU" ./build/bench_fftw "$@"
fi
exec ./build/bench_fftw "$@"
