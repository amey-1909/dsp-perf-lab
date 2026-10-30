#!/usr/bin/env bash
# Ask GCC why the scalar reference loops were / were not auto-vectorized.
# This output is the raw material for the "why" section of your write-up.
set -euo pipefail
cd "$(dirname "$0")/.."
for f in kernels/autovec/fir_f32_autovec.c kernels/autovec/gemm_s8_autovec.c; do
  echo "=== $f ==="
  gcc -O3 -mavx2 -mfma -Iinclude -fopt-info-vec-optimized -fopt-info-vec-missed \
      -c "$f" -o /dev/null 2>&1 | grep -E 'ref\.c' | sort -u | head -25
done
