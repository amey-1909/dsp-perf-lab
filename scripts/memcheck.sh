#!/usr/bin/env bash
# Valgrind memcheck over the test-suite (use a non-sanitized Release/Debug build).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j >/dev/null
valgrind --error-exitcode=1 --leak-check=full ./build/test_kernels
