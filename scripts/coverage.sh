#!/usr/bin/env bash
# gcov/lcov line + branch coverage of the kernels.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DDSP_COVERAGE=ON >/dev/null
cmake --build build-cov -j >/dev/null
(cd build-cov && ctest --output-on-failure)
lcov --capture --directory build-cov --output-file build-cov/cov.info \
     --rc branch_coverage=1 --ignore-errors mismatch,inconsistent >/dev/null
lcov --extract build-cov/cov.info "$PWD/kernels/*" --output-file build-cov/kernels.info \
     --rc branch_coverage=1 >/dev/null
lcov --list build-cov/kernels.info --rc branch_coverage=1
