# AVX2 FFT: a measured optimization ladder

Single-precision, interleaved complex, radix-2 DIT FFT on x86-64 with AVX2 + FMA.
Every step changes one thing and is checked for correctness and timed against the
scalar reference and against the previous step. Every claim below comes with the
hardware counter that explains it.

**Result:** 2.9x faster than the scalar reference at n = 256, 2.7x at n = 1024,
2.6x at n = 4096, 2.1x at n = 16384 and 2.2x at n = 65536. Against FFTW 3 (single
precision, same conditions) it reaches 31-44% of FFTW's speed, i.e. FFTW is 2.3-3.2x faster.

## Test machine

| | |
|---|---|
| CPU | Intel Core i7-10510U @ 1.80 GHz (Comet Lake, Skylake-derived core; 4 cores / 8 threads, AVX2 + FMA, no AVX-512) |
| Caches | L1d 32 KB per core, L2 256 KB per core, L3 8 MB shared (`lscpu` totals 128 KiB / 1 MiB / 8 MiB divided by 4 cores) |
| Compiler | GCC 9.4.0 (Ubuntu 20.04), `-O3 -g`; the AVX2 sources are compiled with `-mavx2 -mfma` |
| Method | pinned with `taskset -c 2`; `bench_fft` times all steps in interleaved rounds; run-to-run noise about ±3% |
| Date | 2026-11-12 |

Working-set sizes on this machine (complex float = 8 bytes): n = 1024 is 8 KB of data and fits
in L1; n = 4096 is 32 KB, which already fills L1d once the twiddle tables are added; n = 65536
is 512 KB of data (plus about the same again in twiddle tables), which exceeds the 256 KB L2
and lives in L3.

## The ladder

| Step | File | What changed |
|---|---|---|
| ref | `kernels/ref/fft_f32_ref.c` | Scalar iterative radix-2 DIT with a precomputed plan (bit-reversal table, twiddles) |
| s1 | `fft_s1_gather.c` | AVX2 baseline: gather loads, gather twiddles, scalar stores |
| s2 | `fft_s2_vector_store.c` | Contiguous loads + shuffles to de-interleave re/im, vector stores through unpack + permute |
| s3 | `fft_s3_twiddle_table.c` | Per-stage contiguous twiddle table (no gathers); loop order swapped so each twiddle vector is loaded once per chunk and reused across all blocks |
| s4 | `fft_s4_direct8.c` | First three stages (half = 1, 2, 4) replaced by one direct 8x8 DFT per group of 8, held in registers |
| s5 | `fft_s5_blocked.c` | Cache blocking: stages up to `BLOCK = 4096` points run block by block so the block stays in cache |

Note: s3 contains two changes (the twiddle table and the loop swap). The twiddle table
alone is what removed the gathers; the loop swap then cut the twiddle reloads. They were
committed together, so the "vs previous" figure for s3 covers both.

## Time per FFT (µs) and speedup over the scalar reference

| n | ref | s1 | s2 | s3 | s4 | s5 |
|--:|--:|--:|--:|--:|--:|--:|
| 256 | 2.16 | 6.08 (0.36x) | 3.73 (0.58x) | 1.46 (1.49x) | 0.74 (2.93x) | 0.74 (2.93x) |
| 1024 | 10.20 | 29.70 (0.34x) | 17.38 (0.59x) | 6.65 (1.53x) | 3.72 (2.75x) | 3.73 (2.74x) |
| 4096 | 47.88 | 144.27 (0.33x) | 82.17 (0.58x) | 29.48 (1.62x) | 18.72 (2.56x) | 18.71 (2.56x) |
| 16384 | 265.11 | 710.27 (0.37x) | 420.41 (0.63x) | 187.72 (1.41x) | 146.97 (1.80x) | 126.36 (2.10x) |
| 65536 | 1377.88 | 3378.12 (0.41x) | 1789.70 (0.77x) | 974.83 (1.41x) | 831.58 (1.66x) | 629.67 (2.19x) |

## Hardware counters per FFT

`scripts/perf_fft.sh`: counters from two runs (K and 2K iterations) are subtracted so that
start-up work cancels out.

**n = 1024** (fits in L1)

| step | instructions | cycles | IPC |
|---|--:|--:|--:|
| ref | 173,628 | 44,022 | 3.94 |
| s1 | 182,611 | 127,695 | 1.43 |
| s2 | 155,812 | 74,329 | 2.10 |
| s3 | 91,268 | 28,451 | 3.21 |
| s4 | 45,932 | 16,303 | 2.82 |
| s5 | 45,956 | 16,100 | 2.85 |

**n = 65536** (512 KB of data: exceeds L1 and the 256 KB L2, so it lives in L3)

| step | instructions | cycles | IPC | L1 load misses |
|---|--:|--:|--:|--:|
| ref | 16,809,759 | 5,734,801 | 2.93 | 688,668 |
| s1 | 13,177,044 | 14,063,692 | 0.94 | 646,171 |
| s2 | 11,225,024 | 7,764,470 | 1.45 | 660,850 |
| s3 | 6,926,385 | 4,263,873 | 1.62 | 383,347 |
| s4 | 4,050,565 | 3,470,062 | 1.17 | 363,303 |
| s5 | 4,207,725 | 2,527,262 | 1.66 | 195,571 |

Topdown level 1 (busy core, `perf stat -a --topdown`, n = 65536):

| step | retiring | bad speculation | frontend bound | backend bound |
|---|--:|--:|--:|--:|
| s3 | 44.3% | 0.8% | 1.4% | 53.4% |
| s4 | 32.0% | 0.7% | 1.7% | 65.6% |
| s5 | 43.1% | 1.1% | 1.4% | 54.4% |

At n = 4096, s4 and s5 (one block, same code path): retiring 67%, backend bound 27%.

## Comparison with FFTW

`scripts/bench_fftw.sh`: single-precision complex-to-complex, in place, forward transform,
32-byte aligned buffer, one memcpy of the input per FFT, pinned to one core, all
implementations timed in interleaved rounds. FFTW is planned with `FFTW_MEASURE` (normal
usage); `FFTW_ESTIMATE` is within 1-2%. Planning time is not included. Relative error against
the scalar reference is about 1e-7 for every implementation.

| n | scalar ref | ours (s5) | FFTW (MEASURE) | ours / FFTW |
|--:|--:|--:|--:|--:|
| 256 | 2.15 µs | 0.74 µs | 0.33 µs | 2.26x slower |
| 1024 | 10.18 µs | 3.67 µs | 1.14 µs | 3.21x slower |
| 4096 | 47.86 µs | 18.96 µs | 6.09 µs | 3.11x slower |
| 16384 | 263.12 µs | 125.69 µs | 40.65 µs | 3.09x slower |
| 65536 | 1358.61 µs | 605.20 µs | 225.03 µs | 2.69x slower |

FFTW is 6-9x faster than the scalar reference; this implementation is 2.1-2.9x faster.

**What FFTW does at n = 1024** (`fftwf_print_plan`): it splits 1024 = 32 x 32 and runs two
passes of unrolled 32-point AVX codelets (`n2fv_32_avx`, then the twiddle codelet
`t2fv_32_avx`), working on a buffer. This implementation makes eight passes over the data
(one direct-8 pass and seven radix-2 passes), each loading, shuffling and storing everything.

**Arithmetic count.** `fftwf_flops` reports 2968 adds, 912 multiplies and 256 FMAs for
n = 1024. These count SIMD operations, so with an AVX width of 8 that is about 35,100 real
flops (an estimate: the factor of 8 is inferred because it matches the split-radix count of
about 34,800). The radix-2 convention used for the GFLOP/s column is 5 n log2 n = 51,200, so
FFTW needs about 31% fewer operations. That explains roughly 1.46x of the 3.21x gap at
n = 1024; the remaining roughly 2.2x is implementation (fewer passes, no per-stage shuffles,
unrolled codelets). This split is approximate.

## What each step bought, and why

**The rule used throughout:** speedup = (instruction-count ratio) x (IPC ratio). Looking
at either alone is misleading.

**s1: gathers cost cycles, not instructions.** The gather version executes about 5% more
instructions than the scalar reference (182.6k vs 173.6k at n = 1024), yet it is 2.9x
slower. A `vgatherdps` is one instruction but many micro-ops, so an instruction counter
hides its cost. IPC fell from 3.94 to 1.43. `perf annotate` put about 30% of the time in
the two twiddle gathers.

**s2: removing the scalar stores.** Slightly fewer instructions (-15%), 1.7x faster than s1:
IPC rose from 1.43 to 2.10 once the 8-element scalar store loop became two vector stores.

**s3: no gathers.** Instructions fell 41% (155.8k to 91.3k) and IPC rose from 2.10 to 3.21
at n = 1024. This is the first step that beats the reference (1.5x).

**s4: fewer instructions, but the stalls show up.** Direct 8x8 instead of three scalar
stages cuts instructions another 50% at n = 1024. On paper the direct form does *more*
arithmetic: 64 complex multiply-adds per group of 8 against about 12 butterflies. It wins
because one FMA computes all 8 outputs at once, with constants in a table and no per-stage
data movement. At n = 65536 the gain is smaller than the instruction count suggests: IPC
dropped from 1.62 to 1.17 and the backend-bound share rose from 53% to 66%. Once the compute
shrank, memory latency that used to hide behind it became visible.

**s5: blocking recovers the stalls.** The first twelve stages (len up to 4096, half up to 2048) touch only one
block at a time, so it stays in cache. At n = 65536: L1 misses -46% (363k to 196k per FFT),
IPC 1.17 to 1.66, backend bound 65.6% to 54.4%, 1.32x faster than s4. Below the block size
s5 equals s4. `BLOCK = 4096` beat `1024` on this CPU (1.32x vs 1.19x at n = 65536). A 4096-point block is
32 KB of data plus about 32 KB of twiddles, which spills L1d (32 KB) but fits easily in the
256 KB L2; most likely the saving from fewer full sweeps outweighs the extra L1 misses.

## Things that went wrong, and what they taught

- **Instruction count versus time.** At n = 1024, s4 executes 3.8x fewer instructions than
  the reference but is only 2.7x faster. The reference runs at IPC 3.94, close to a 4-wide
  limit, so cutting instructions can only buy so much.
- **A real regression, found by counters.** The first version of s5 was 7-15% *slower* than
  s4 at n <= 4096 even though one block does identical work. `perf stat` showed 0.6% more
  instructions but IPC 2.86 -> 2.51: a code-generation effect of splitting the stage loop
  into helper functions, not an algorithmic one. TODO: state the fix that was applied
  (small sizes dispatched to the s4 path / forced inlining) and the result.
- **A measurement artifact, caught by cross-checking.** `perf stat -a -C 2 --topdown`
  reported retiring above 100% and negative backend-bound. With hyperthreading, a
  per-logical-CPU measurement divides by half the core's slots. Retiring should be about
  IPC / 4; every affected run read about double. Using core-level counting (`-a` without
  `-C`) gave numbers that match IPC / 4.
- **Bit-exactness.** From s4 on, the output is not bit-identical to the previous step (different
  rounding from constants and FMA order). Correctness is checked by relative L2 error
  (about 1e-7 for all steps) against a naive DFT and against NumPy.

## Correctness

- C test suite: every registered step is checked against a golden-model DFT for
  n = 2 .. 8192 (7,435 checks).
- `python/validate.py`: NumPy FFT cross-check for every step.
- Clean under ASan, UBSan and Valgrind.

## Reproduce

```bash
cmake -S . -B build && cmake --build build -j
scripts/fft_check.sh 256 1024 4096 16384 65536     # tests + time ladder
python3 python/validate.py build/libdsp.so          # NumPy cross-check
scripts/perf_fft.sh                                 # instructions / cycles / IPC / L1 misses per step
scripts/bench_fftw.sh                               # comparison with FFTW (needs libfftw3-dev)
sudo perf stat -a --topdown taskset -c 2 ./build/fft_run s5 65536 5000   # topdown, read the busy core's row
```

## What I would do next

- **Fewer passes** (the main gap to FFTW: 2 passes against 8 at n = 1024). Radix-4 takes the
  seven radix-2 passes to about four, with 3 complex multiplies instead of 4 per group and half
  the loads, stores and shuffles; radix-8 or 16 goes further at the cost of register pressure
  (16 ymm registers). At s5 the backend is still about 54% busy waiting at n = 65536, so
  fewer sweeps should help.
- **Split-complex layout** between stages (separate re and im arrays) to remove the
  per-stage de-interleave/interleave shuffles.
- **Bit reversal** (roughly a quarter of the instructions at n = 1024, estimated from a callgrind breakdown): a precomputed
  swap-pair list, or fusing the permutation into the first pass.

## Limitations

One machine, one compiler, single core. Differences under about 3% are noise. Topdown on this
Skylake-derived core only reports level 1, so the backend-bound share cannot be split into memory-bound and
core-bound.
