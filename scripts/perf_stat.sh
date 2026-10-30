#!/usr/bin/env bash
# Hardware counters for the whole benchmark: IPC, cache and branch behaviour.
# Pinned to one core for stable numbers. Needs `perf` and counter access
# (bare metal, or a VM that exposes the PMU; sudo sysctl kernel.perf_event_paranoid=1).
set -euo pipefail
cd "$(dirname "$0")/.."
exec taskset -c "${CPU:-2}" perf stat \
  -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
  ./build/bench "$@"
