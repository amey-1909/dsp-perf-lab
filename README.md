# dsp-perf-lab

Performance engineering for DSP / ML kernels on x86 (AVX2 + FMA). Each kernel is built as a
**ladder**: one optimization per step, every step checked against an independent golden model, and
every step timed against the one below it. The repo shows *how* each speedup was obtained, and
which hardware counter explains it, not just that it exists.

Kernels: FFT (f32), int8 GEMM, FIR, requantization, SAD. The FFT and GEMM ladders are the main work.

## Headline results

All numbers: Intel Core i7-10510U (Comet Lake, 4 cores, L1d 32 KB, L2 256 KB, L3 8 MB, AVX2, no
AVX-512/VNNI), GCC 9.4.0 `-O3`, pinned with `taskset -c 2`, run-to-run noise about ±3%.

| Kernel | Result | Details |
|---|---|---|
| FFT f32 (radix-2, interleaved) | 2.1x to 2.9x over the scalar reference, n = 256 to 65536; 31-44% of FFTW's speed (FFTW is 2.3-3.2x faster) | [docs/fft_results.md](docs/fft_results.md) |
| FFT, split re/im layout (step 6) | **TBD: fill in from `bench_fft` after running it** | below |
| int8 GEMM (bit-exact) | about 21x over the scalar reference (86 GOP/s at 128x128x256), about 100 GOP/s on larger `Bt` | below |

## Quick start

```bash
cmake -S . -B build && cmake --build build -j
./build/test_kernels                            # C tests (7,500+ checks, every ladder step)
python3 python/validate.py build/libdsp.so      # cross-check against NumPy
scripts/fft_check.sh                            # build + test + FFT ladder table
scripts/gemm_check.sh                           # build + test + GEMM ladder table
taskset -c 2 ./build/bench                      # FIR / GEMM / FFT summary table
```

Counter scripts need `sudo sysctl kernel.perf_event_paranoid=1`:

```bash
scripts/perf_fft.sh 1024 65536                  # per FFT step: ns, instructions, cycles, IPC, L1 misses
PORTS=1 scripts/perf_gemm.sh 128 128 256        # per GEMM step: IPC and port 0/1/5 utilization
scripts/bench_fftw.sh                           # compare with FFTW (optional dependency)
```

## Layout

| Path | Contents |
|---|---|
| `include/dsp.h` | Public API; each kernel exists as `_ref`, `_autovec`, `_avx2`; ladder API (`fft_avx2_step_*`, `gemm_avx2_step_*`) |
| `kernels/ref/` | Scalar C, auto-vectorization disabled. The correctness oracle |
| `kernels/autovec/` | Same source, `-O3 -mavx2 -mfma`: what the compiler gives you |
| `kernels/avx2/fft/` | One file per FFT step: `fft_s<N>_<name>.c`, registered in `fft_steps.c` |
| `kernels/avx2/gemm/` | One file per GEMM step: `gemm_s<N>_<name>.c`, registered in `gemm_steps.c` |
| `tests/` | C tests against double-precision / int64 golden models, odd shapes and tails |
| `python/validate.py` | ctypes cross-check against NumPy (`np.correlate`, `np.fft`, int32 matmul) |
| `bench/` | Step-ladder benchmarks (`bench_fft`, `bench_gemm`), single-step runners for `perf` (`fft_run`, `gemm_run`), FFTW comparison |
| `scripts/` | Step scaffolding, check scripts, counter scripts, `vec_report.sh`, `mca.sh`, `memcheck.sh`, `coverage.sh` |

## The ladder method

Every AVX2 optimization lives in its own file and is registered in a table. The C tests, the NumPy
cross-check and the benchmark iterate over that table, so each step gets its own correctness check
and two speedup numbers: against the scalar reference and against the previous step. A step that
gives wrong results is reported and never timed.

```bash
scripts/new_gemm_step.sh <name> "<what you will change>"   # copy the last step to N+1 and register it
$EDITOR kernels/avx2/gemm/gemm_s<N+1>_<name>.c             # change ONE thing
scripts/gemm_check.sh                                      # build, test every step, print the table
```

The same flow exists for FFT (`scripts/new_fft_step.sh`, `scripts/fft_check.sh`). The public
`gemm_s8_avx2()` always runs the newest GEMM step.

Benchmarks time every function in interleaved rounds and keep each one's best round, so slow drift
does not favor whichever function ran during a bad stretch. Differences inside about ±3% are noise.

## FFT ladder

Single-precision, interleaved complex, radix-2 decimation-in-time. Full write-up with counters:
[docs/fft_results.md](docs/fft_results.md).

| Step | File | What changed |
|---|---|---|
| ref | `kernels/ref/fft_f32_ref.c` | Scalar iterative radix-2 with a precomputed plan |
| s1 | `fft_s1_gather.c` | AVX2 baseline: gather loads, gather twiddles, scalar stores |
| s2 | `fft_s2_vector_store.c` | Contiguous loads + shuffles to de-interleave, vector stores |
| s3 | `fft_s3_twiddle_table.c` | Per-stage contiguous twiddle table, loop swap (no gathers, twiddles loaded once per chunk) |
| s4 | `fft_s4_direct8.c` | First three stages as one direct 8x8 DFT per group, in registers |
| s5 | `fft_s5_blocked.c` | Cache blocking: stages up to 4096 points run block by block |
| s6 | `fft_s6_split.c` | Split layout: separate re/im arrays, no de/interleave shuffles in the stages; `fft_f32_avx2_split()` for data that is already split |

Time per FFT in µs (speedup over the scalar reference), steps s1-s5:

| n | ref | s1 | s3 | s4 | s5 |
|--:|--:|--:|--:|--:|--:|
| 256 | 2.16 | 6.08 (0.36x) | 1.46 (1.49x) | 0.74 (2.93x) | 0.74 (2.93x) |
| 1024 | 10.20 | 29.70 (0.34x) | 6.65 (1.53x) | 3.72 (2.75x) | 3.73 (2.74x) |
| 4096 | 47.88 | 144.27 (0.33x) | 29.48 (1.62x) | 18.72 (2.56x) | 18.71 (2.56x) |
| 16384 | 265.11 | 710.27 (0.37x) | 187.72 (1.41x) | 146.97 (1.80x) | 126.36 (2.10x) |
| 65536 | 1377.88 | 3378.12 (0.41x) | 974.83 (1.41x) | 831.58 (1.66x) | 629.67 (2.19x) |

Step 6 (`s6_split` and the `split (native)` row in `bench_fft`): **TBD, add measured numbers here.**
`s6_split` takes interleaved data and pays two conversion passes; `split (native)` is the same FFT on
data that is already held as separate arrays.

What the steps showed:

- **Gathers are one instruction but many µops.** Instruction counts alone hid why s1 was slower
  than scalar; the counters did not.
- **Fewer, bigger passes win.** The direct 8x8 DFT replaced three stages with one register-resident
  step (s4). Speedup is the instruction-count ratio times the IPC ratio, so both were measured.
- **Blocking only helps once the data outgrows the cache.** s5 is flat up to n = 4096 and gains
  16% at n = 16384 and 32% at n = 65536, where a transform no longer fits in L1/L2.
- **FFTW gap.** FFTW needs about 2 passes at n = 1024 where this code needs 8, and uses about 31%
  fewer arithmetic operations through fused codelets. That is a hypothesis, partly measured.

## int8 GEMM ladder

`C[M][N] (int32) = A[M][K] (int8) x B`, with B passed pre-transposed as `Bt[N][K]` so both operands
stream along K. Bit-exact against an int64 golden model that uses the original B, so the layout
convention is validated too. GOP/s counts 2 operations per multiply-accumulate. VNNI and `sdot`
are not available on this CPU, so the kernels use sign-extend (`vpmovsxbw`) + `vpmaddwd`, which
is exact for signed x signed and cannot overflow.

| Step | File | What changed |
|---|---|---|
| ref | `kernels/ref/gemm_s8_ref.c` | Scalar triple loop |
| autovec | | Same source compiled with `-O3 -mavx2 -mfma` |
| s1 | `gemm_s1_widen_each_time.c` | 1 row x 4 columns, widen operands in the inner loop |
| s2 | `gemm_s2_tile2x4.c` | 2 x 4 register tile: A widened once per 4 columns, Bt once per 2 rows |
| s3 | `gemm_s3_prepack16.c` | Bt widened to int16 once up front (padded, 32-byte-aligned rows, aligned loads) |
| s4 | `gemm_s4_block_l2.c` | Column blocking: a 128 KB int16 chunk of Bt widened into a fixed buffer, reused by every row pair |

**128 x 128 x 256** (GOP/s, speedup over the scalar reference):

| ref | autovec | s1 | s2 | s3 | s4 |
|--:|--:|--:|--:|--:|--:|
| 4.09 (1.00x) | 22.12 (5.4x) | 77.21 (18.9x) | 85.89 (21.0x) | 88.54 (21.7x) | 86.24 (21.1x) |

**Bt footprint sweep** (M = 64, K = 1024, GOP/s; `Bt` is the int8 size, N x K bytes):

| Bt (int8) | 16 KB | 256 KB | 1 MB | 4 MB |
|---|--:|--:|--:|--:|
| s2 | 99.7 | 98.9 | 98.7 | 97.8 |
| s3 | 97.2 | 84.1 | 84.0 | 37.0 |
| s4 | 97.1 | 95.1 | 95.7 | 98.1 |

What the steps showed:

- **s1 to s2 (register tile): +11-28%.** Sharing the widened loads across a 2 x 4 tile reduces sign-extensions
  per MAC by about 40%. The tile also halves the Bt bytes streamed per MAC, which is why s2 stays flat from
  16 KB to 4 MB while s1 loses 21% at 4 MB.
- **s3 (pre-widening) alone does not pay on this CPU.** The port-5 counter fell from 93% to 79%, but cycles did not,
  because the int16 copy doubles the footprint: 14% slower once it passes the 256 KB L2, and a cliff
  (37 GOP/s) when 8 MB of int16 `Bt` equals the whole L3.
- **A compiler detail cost 20% of the instructions.** GCC 9 splits unaligned 256-bit loads into
  `vmovdqu xmm` + `vinserti128`. Padding rows to 32 bytes and using aligned loads removed it
  (instructions per GEMM 1.47M to 1.18M).
- **s4 (blocking) fixes s3's loss.** The cliff goes away (37 to 98 GOP/s) and the curve is flat, at about the speed of s2. On this CPU the
  result of s3 + s4 is "same speed as s2 with a bounded 128 KB buffer": pre-widening only pays off when
  combined with other changes. That is a measured negative result, kept in the ladder on purpose.
- **Counters read with care.** `uops_dispatched_port.port_5` also counts `vpaddd`, so "port 5 busy" is not the same as
  "widening-bound". The hot loop is instruction-count bound (IPC about 3 on a 4-wide core): 38 instructions
  per tile step against a minimum of about 24, because GCC adds accumulator copies and a stack spill.
- **Cache blocking was added only where a sweep showed a drop.** s2 needed none. K and M blocking were not
  needed at these shapes and are untested.

Next for GEMM: remove the accumulator copies (2 x 3 or 3 x 4 tile, loop unrolling), a k-pair packed layout that
avoids the horizontal sums, fused requantization, an AVX-512 VNNI path with runtime dispatch, and an ARM NEON port.

## Testing

- Golden models are independent of the kernels: double precision for float, int64 for integer, a naive O(n^2) DFT for FFT.
- Every registered ladder step is tested on its own, on awkward shapes (n = 1, odd M/N, K not a multiple of 16).
- GEMM steps are also run into a buffer with guard words around `C`, which catches out-of-bounds writes
  (it caught a heap overflow in a first 2-row version).
- Buffers are exact-size so ASan/Valgrind catch overruns: `cmake -S . -B build-san -DDSP_SANITIZE=ON`.

## Other kernels

| Kernel | ref | autovec | AVX2 | Notes |
|---|:-:|:-:|:-:|---|
| FIR f32 | done | done | done | Vectorized across output samples with 4 accumulators and one tap broadcast reused four times; `-O3` cannot reassociate the float reduction without `-ffast-math` |
| Requantization (Q31 multiplier + shift) | done | | | Bit-exact gemmlowp/TFLite arithmetic, the step an int8 NPU does after its MAC array |
| SAD 16x16 u8 (+ `sad_x4`) | done | done | done | Experiments in `bench/bench_sad.c`: load-bound check, working-set sweep, stride conflicts. The sample results were measured on a different machine (cloud Xeon VM), so re-run before quoting them |

## Profiling workflow

```bash
scripts/vec_report.sh        # why did / didn't GCC vectorize the reference loops?
scripts/perf_stat.sh         # IPC, cache-miss and branch-miss
scripts/mca.sh skylake       # llvm-mca: predicted IPC and port pressure
scripts/memcheck.sh          # valgrind
scripts/coverage.sh          # gcov/lcov
```

Hardware counters do not work in most cloud VMs. Run them on bare metal.

Measuring one step: `./build/fft_run s4 1024 100000` and `./build/gemm_run s2 128 128 256 20000` run a single
implementation many times (after checking it) so `perf stat` counters belong to that step alone.
The counter scripts subtract a K-iteration run from a 2K-iteration run so start-up cost cancels.

Top-down caveat on SMT machines: per-CPU top-down figures can be about 2x wrong. Use `perf stat -a --topdown`
without `-C`, read the busy core, and cross-check with retiring about IPC / 4.

## Roadmap

1. **GEMM**: tile/loop codegen (3 x 4 or 2 x 3, unrolling), packed k-pair layout, fused requant, VNNI path, NEON port.
2. **FFT**: radix-4 (fewer passes, the main gap to FFTW), out-of-place split API, Q15 fixed point.
3. **Multithreading** (OpenMP): scaling plot, false sharing and NUMA discussion.
4. **ARM NEON port** of the GEMM and FFT inner loops (`sdot`, `vaddv`).
5. **GPU port** (CUDA/OpenCL) with an honest look at where transfer cost loses.
6. **DSP port**: Hexagon SDK simulator (HVX) or TI C6x simulator; cycle counts and software pipelining.
7. **Wake-word pipeline**: mel filterbank, MFCC and a DS-CNN built from these operators.

## Reporting results

Numbers depend on CPU, frequency scaling and noisy neighbours. Before quoting one (README, resume, interview), re-run
the benchmark pinned to one core on your own hardware and record the CPU model and compiler version next to it.
