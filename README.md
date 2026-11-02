# dsp-perf-lab

A small performance-engineering lab for DSP / ML kernels on x86. Every kernel
is kept at **every stage of the optimization ladder**, tested against an
independent golden model, and benchmarked against the stage below it, so the
repo shows *how* the speedup was obtained, not just that it exists.

Targets the skill set of HPC / processor-level performance roles: SIMD,
micro-architecture reasoning, fixed-point DSP, int8 NN operators, profiling.

## Quick start

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure      # C tests (7,000+ checks)
python3 python/validate.py build/libdsp.so      # cross-check vs NumPy
taskset -c 2 ./build/bench                      # benchmark table (FIR, GEMM, FFT)
taskset -c 2 ./build/bench_sad                  # SAD experiments
scripts/fft_check.sh                            # build + test + FFT step-ladder table
```

## Layout

| Path | Contents |
|---|---|
| `include/dsp.h` | Public API; every kernel exists as `_ref`, `_autovec`, `_avx2` |
| `kernels/ref/` | Scalar C, auto-vectorization disabled. The correctness oracle |
| `kernels/autovec/` | Same source, `-O3 -mavx2 -mfma`: what the compiler gives you |
| `kernels/avx2/` | Hand-written AVX2/FMA intrinsics; `fft/` holds the FFT step ladder |
| `tests/` | C tests vs double-precision / int64 golden models, awkward tail shapes |
| `python/validate.py` | ctypes cross-check against NumPy (`np.correlate`, `np.fft`, int32 matmul) |
| `bench/` | `bench.c`: kernel ladder table. `bench_sad.c`: SAD load-bound / cache experiments |
| `scripts/` | `vec_report.sh`, `perf_stat.sh`, `mca.sh`, `memcheck.sh`, `coverage.sh` |

## Kernels and ladder status

| Kernel | ref | autovec | AVX2 | Cache/layout | Threads | GPU | DSP sim |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| FIR f32 | done | done | done | | | | |
| int8 GEMM (B pre-transposed) | done | done | done | | | | |
| Requantization (Q31 mult + shift) | done | | | | | | |
| Radix-2 FFT f32 | done | | step 1 (see FFT step ladder) | | | | |
| SAD 16x16 u8 (+ `sad_x4`) | done | done | done | experiments only | | | |

Next rungs, in suggested order: see **Roadmap**.

## Design notes

**FIR (`fir_f32_avx2.c`).** The scalar loop is a float reduction over taps.
Without `-ffast-math`, GCC may not reassociate it: `scripts/vec_report.sh`
shows it vectorizes the multiplies but keeps the adds in order, so the gain is
small. The AVX2 version vectorizes across *output samples* instead, keeping
per-output tap order, with 4 independent accumulators to hide FMA latency and
one broadcast of `h[k]` reused four times.

**int8 GEMM (`gemm_s8_avx2.c`).** `cvtepi8_epi16` + `madd_epi16` is exact for
signed x signed and cannot overflow (max pair sum 32768), so results are
bit-identical to the reference. Register-blocked 1 row x 4 columns so the
widened A vector is reused. B is passed transposed so both operands stream
along K, which is the layout dot-product hardware (VNNI, NEON `sdot`, Hexagon
HVX, NPUs) wants.

**Requantization.** Bit-exact gemmlowp/TFLite arithmetic
(`SaturatingRoundingDoublingHighMul` + `RoundingDivideByPOT`): the step every
int8 NPU pipeline performs after its MAC array.

**Testing approach.** Golden models are independent of the kernels (double
precision for float, int64 for integer, naive O(n^2) DFT for FFT). The GEMM
golden uses the original non-transposed B, so it validates the layout
convention too. Buffers are exact-size so ASan/Valgrind catch overruns.

## FFT step ladder (one optimization per step)

Each AVX2 FFT optimization lives in its own file,
`kernels/avx2/fft/fft_s<N>_<name>.c`, as the function `fft_f32_avx2_s<N>`. Steps
are registered in `kernels/avx2/fft/fft_steps.c`; the C tests, the NumPy
cross-check and `bench_fft` iterate over that table. So every step gets its own
correctness check, and its own speedup number both against the scalar reference
and against the previous step. The second number tells you exactly what a
change bought. A step that produces wrong results is reported and never timed.

| Step | File | What changed |
|---|---|---|
| 1 | `fft_s1_gather.c` | Baseline AVX2: gather loads, gather twiddles, scalar stores |

Workflow for each new step:

```bash
scripts/new_fft_step.sh <name> "<what you will change>"   # copies the latest step into step N+1 and registers it
$EDITOR kernels/avx2/fft/fft_s2_<name>.c                  # change ONE thing
scripts/fft_check.sh                                      # build, test every step, print the ladder table
git add -A && git commit -m "FFT step 2: <what changed and the speedup>"
```

Add a row to the table above for each step, with the measured `vs prev` figure.
`scripts/fft_check.sh 1024 8192` benchmarks chosen sizes.

**Profiling one step.** `./build/fft_run <ref|s1|s2...> <n> <iters>` runs a single
implementation many times (after checking it against the reference), so
`perf stat` counters belong to that step alone:

```bash
taskset -c 2 perf stat -e cycles,instructions,cache-misses ./build/fft_run s1 1024 100000
```

Use 100000 or more iterations; 1000 FFTs take only ~10-30 ms and give noisy figures.

**IPC for FIR / GEMM / SAD.** `scripts/perf_kernels.sh [fir|gemm|sad|sadx4 ...]` runs each
variant (ref, autovec, avx2) alone under `perf stat` and prints ns, instructions and cycles per
unit of work plus IPC. Needs `sudo sysctl kernel.perf_event_paranoid=1`. Read *instructions per
unit* first: a SIMD kernel often has lower IPC than scalar and is still much faster.

**Noise floor.** `bench_fft` times all steps in interleaved rounds to remove
drift, but small differences are still noise. On a shared cloud VM, two
byte-identical steps read 0.97x to 1.00x "vs prev" (about +/-3%), so treat
anything inside that band as no change, and re-run a few times before
believing a small gain. A quiet bare-metal machine will be tighter.

## SAD experiments (video motion estimation)

`bench/bench_sad.c` asks one question three ways: is a 16x16 SAD load-bound,
cache-bound, or neither?

| Experiment | What it varies | What it shows |
|---|---|---|
| E1 load-bound check | single SAD vs `sad_x4` (current block loaded once for 4 candidates), data all in L1 | Whether reusing loads helps. Includes a measured **empty-call control**, because x4 makes one call per four SADs |
| E2 working-set sweep | frame size 4 KB to 64 MB, SADs at random positions | Where ns/SAD steps up as the data outgrows L1, L2, L3 |
| E3 stride | same block, same work, row stride 1024/2048/4096/8192 vs the same +64 | Conflict misses from power-of-two strides |

**Sample run** (shared cloud Xeon VM, L1D 48 KB 12-way, L2 2 MB; numbers differ
on other machines, so re-run before quoting anything):

- *E1:* GCC's auto-vectorized scalar loop is as fast as the hand-written AVX2
  single SAD, so the intrinsics add nothing there. `sad_x4` looked 1.58x faster
  per SAD, but an empty library call cost about 3 ns of a ~7.8 ns SAD; adjusted
  for it, the kernel-level gain is about **1.14x**. Most of the raw gain was call
  overhead, not load reuse.
- *E2:* about 8 ns/SAD while the frame fits L1; 1.1x to 1.8x slower up to 1 MB
  (L2 hits); about 4x at 4 MB (past L2); about 11x at 64 MB. Random positions are
  the worst case for locality: a raster-order search window is expected to stay
  L1-resident (not measured here).
- *E3:* strides of 4096 and 8192 were consistently ~15% slower than the same
  stride + 64, while 1024 and 2048 showed no effect. That onset matches 16 rows
  overflowing 12 ways at 4096-byte spacing, but it is inferred from timing only;
  confirm with L1 miss counters on bare metal.

Conclusion for this machine: a single SAD on L1-resident data is **not**
cache-bound; it becomes memory-bound when positions scatter across a frame
larger than L2.

## Profiling workflow

```bash
scripts/vec_report.sh        # why did / didn't GCC vectorize the reference loops?
scripts/perf_stat.sh         # IPC, cache-miss and branch-miss (needs PMU access)
scripts/mca.sh skylake       # llvm-mca: predicted IPC and port pressure of the GEMM inner loop
scripts/memcheck.sh          # valgrind
scripts/coverage.sh          # gcov/lcov
cmake -S . -B build-san -DDSP_SANITIZE=ON   # ASan + UBSan
```

Note: `perf stat` hardware counters usually do not work inside cloud VMs or
containers. Run it on bare metal or a VM that exposes the PMU.

## Roadmap

1. **FFT**: continue the step ladder (below), one change per step; then radix-4
   and Q15 fixed-point.
2. **SIMD requant**: AVX2 `srdhm` via `_mm256_mul_epi32` + shuffle; fuse it into
   the GEMM epilogue.
3. **GEMM cache blocking and 2x4 register tile**; prefetching; measure against
   a roofline (peak from `perf` counters, bandwidth from a STREAM run).
4. **AVX-512 VNNI** path (`vpdpbusd`) for CPUs that have it, with runtime dispatch.
5. **Multithreading** (OpenMP): scaling plot, false-sharing and NUMA discussion.
6. **GPU port** (CUDA/OpenCL) of FFT and GEMM and an honest analysis of where
   transfer cost makes the GPU lose.
7. **DSP port**: Hexagon SDK simulator (HVX) or TI CCS C6x simulator for the
   FIR and GEMM; cycle counts and a software-pipelining write-up.
8. **DMA/TCM model**: ping-pong double-buffered tiling around a mock DMA.
9. **Wake-word pipeline**: mel filterbank, MFCC and a DS-CNN built from these
   operators, with an end-to-end latency demo.

## Reporting results

Benchmark numbers depend heavily on CPU, frequency scaling and noisy
neighbours. Before quoting any number (README, resume, interview), re-run
`./build/bench` pinned to one core on your own hardware, and record the CPU
model and compiler version next to it.
