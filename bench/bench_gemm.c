/*
 * GEMM step-ladder benchmark: ref, autovec, then every AVX2 step in order, with
 * GOP/s (2 ops per MAC) and speedup vs the scalar ref and vs the PREVIOUS step.
 * Each step is checked bit-exact against the reference first; a wrong step is
 * reported and not timed. Timing is interleaved rounds, best round kept.
 *
 *   ./build/bench_gemm                    # 128x128x256  256^3  64x512x512
 *   ./build/bench_gemm 1024 1024 1024     # one chosen shape (M N K)
 *   taskset -c 2 ./build/bench_gemm       # pin for stable numbers
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUNDS 15
#define MAXF 64

static uint32_t rng = 777u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static void run_shape(int M, int N, int K)
{
    const double macs = (double)M * N * K;
    int8_t *A = malloc((size_t)M * K), *Bt = malloc((size_t)N * K);
    int32_t *C = malloc(sizeof(int32_t) * (size_t)M * N), *R = malloc(sizeof(int32_t) * (size_t)M * N);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd();
    for (size_t i = 0; i < (size_t)N * K; i++) Bt[i] = (int8_t)rnd();
    gemm_s8_ref(A, Bt, R, M, N, K);

    gemm_fn fn[MAXF]; const char *nm[MAXF]; int ok[MAXF]; double best[MAXF];
    int nf = 0;
    fn[nf] = gemm_s8_ref;     nm[nf++] = "ref";
    fn[nf] = gemm_s8_autovec; nm[nf++] = "autovec";
    const int has = dsp_cpu_has_avx2();
    for (int s = 0; has && s < gemm_avx2_num_steps() && nf < MAXF; s++) {
        fn[nf] = gemm_avx2_step_fn(s); nm[nf++] = gemm_avx2_step_name(s);
    }
    for (int f = 0; f < nf; f++) {
        memset(C, 0x5a, sizeof(int32_t) * (size_t)M * N);
        fn[f](A, Bt, C, M, N, K);
        ok[f] = memcmp(C, R, sizeof(int32_t) * (size_t)M * N) == 0;
        best[f] = 1e300;
    }
    /* iterations so one batch is ~20 ms for the ref-speed-independent fast case */
    long it = 1;
    for (;;) {
        double t0 = now_ns();
        for (long i = 0; i < it; i++) fn[nf - 1](A, Bt, C, M, N, K);
        if (now_ns() - t0 > 2e7) break;
        it *= 2;
    }
    for (int r = 0; r < ROUNDS; r++)
        for (int q = 0; q < nf; q++) {
            int f = (q + r) % nf;
            if (!ok[f]) continue;
            long n = f == 0 ? (it >= 8 ? it / 8 : 1) : it;   /* ref is much slower */
            double t0 = now_ns();
            for (long i = 0; i < n; i++) fn[f](A, Bt, C, M, N, K);
            double t = (now_ns() - t0) / (double)n;
            if (t < best[f]) best[f] = t;
        }
    printf("\n%dx%dx%d  (A %.0f KB, Bt %.0f KB)\n", M, N, K, M * (double)K / 1024, N * (double)K / 1024);
    printf("| step | us | GOP/s | vs ref | vs prev | check |\n|------|---:|------:|-------:|--------:|-------|\n");
    double prev = 0;
    for (int f = 0; f < nf; f++) {
        if (!ok[f]) { printf("| %s | - | - | - | - | WRONG |\n", nm[f]); continue; }
        printf("| %s | %.2f | %.2f | %.2fx | %s%.2fx | ok |\n", nm[f], best[f] / 1e3, 2 * macs / best[f],
               best[0] / best[f], "", prev > 0 ? prev / best[f] : 1.0);
        if (f >= 1) prev = best[f];
    }
    free(A); free(Bt); free(C); free(R);
}

int main(int argc, char **argv)
{
    if (argc == 4) { run_shape(atoi(argv[1]), atoi(argv[2]), atoi(argv[3])); return 0; }
    run_shape(128, 128, 256);
    run_shape(256, 256, 256);
    run_shape(64, 512, 512);
    return 0;
}
