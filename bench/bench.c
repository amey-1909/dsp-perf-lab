/* Micro-benchmark harness: min-of-N timing, results as a markdown table.
 *
 * Method: calibrate an iteration count so one timed batch takes ~20 ms, then
 * take the best of several batches (min filters scheduler/IRQ noise; the
 * kernels are deterministic so the minimum is the right estimator).
 * Pin the process for stable numbers:  taskset -c 2 ./bench
 */
#define _POSIX_C_SOURCE 199309L
#include "dsp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

#define BENCH(result, stmt)                                              \
    do {                                                                 \
        long it_ = 1;                                                    \
        double t_;                                                       \
        for (;;) {                                                       \
            double t0_ = now_ns();                                       \
            for (long i_ = 0; i_ < it_; i_++) { stmt; }                  \
            t_ = now_ns() - t0_;                                         \
            if (t_ > 2e7) break;                                         \
            it_ *= 2;                                                    \
        }                                                                \
        double best_ = t_ / (double)it_;                                 \
        for (int r_ = 0; r_ < 7; r_++) {                                 \
            double t0_ = now_ns();                                       \
            for (long i_ = 0; i_ < it_; i_++) { stmt; }                  \
            double tt_ = (now_ns() - t0_) / (double)it_;                 \
            if (tt_ < best_) best_ = tt_;                                \
        }                                                                \
        (result) = best_;                                                \
    } while (0)

static void row(const char *kernel, const char *variant, double ns,
                double ops, double ref_ns)
{
    printf("| %-22s | %-8s | %10.2f | %9.2f | %7.2fx |\n", kernel, variant,
           ns / 1000.0, ops / ns, ref_ns / ns);
}

static void header(const char *unit)
{
    printf("\n| %-22s | %-8s | %10s | %9s | %8s |\n", "kernel", "variant",
           "time (us)", unit, "speedup");
    printf("|%s|%s|%s|%s|%s|\n", "------------------------", "----------",
           "------------", "-----------", "----------");
}

static uint32_t s = 12345u;
static uint32_t rnd(void) { s = s * 1664525u + 1013904223u; return s >> 8; }

int main(void)
{
    const int avx2 = dsp_cpu_has_avx2();
    printf("# dsp-perf-lab benchmark (AVX2+FMA: %s)\n", avx2 ? "yes" : "NO");
    printf("Speedup is vs the scalar ref (auto-vectorization disabled).\n");

    /* ---- FIR: n outputs, `taps` taps, 2 flops per MAC ------------------ */
    {
        const int n = 1 << 16, taps = 64;
        float *x = malloc(sizeof(float) * (size_t)(n + taps));
        float *h = malloc(sizeof(float) * (size_t)taps);
        float *y = malloc(sizeof(float) * (size_t)n);
        for (int i = 0; i < n + taps; i++) x[i] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        for (int k = 0; k < taps; k++) h[k] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        const double flops = 2.0 * n * taps;

        double t_ref, t_av = 0, t_x2 = 0;
        BENCH(t_ref, fir_f32_ref(x, h, y, n, taps));
        if (avx2) {
            BENCH(t_av, fir_f32_autovec(x, h, y, n, taps));
            BENCH(t_x2, fir_f32_avx2(x, h, y, n, taps));
        }
        char name[64];
        snprintf(name, sizeof name, "FIR f32 n=%d taps=%d", n, taps);
        header("GFLOP/s");
        row(name, "ref", t_ref, flops, t_ref);
        if (avx2) {
            row(name, "autovec", t_av, flops, t_ref);
            row(name, "avx2", t_x2, flops, t_ref);
        }
        free(x); free(h); free(y);
    }

    /* ---- int8 GEMM: 2*M*N*K ops --------------------------------------- */
    {
        const int shapes[][3] = {{128, 128, 256}, {256, 256, 256}, {64, 512, 512}, {1024, 1024, 1024}};        header("GOP/s");
        for (size_t si = 0; si < sizeof shapes / sizeof *shapes; si++) {
            const int M = shapes[si][0], N = shapes[si][1], K = shapes[si][2];
            int8_t *A = malloc((size_t)M * K), *Bt = malloc((size_t)N * K);
            int32_t *C = malloc(sizeof(int32_t) * (size_t)M * N);
            for (int i = 0; i < M * K; i++) A[i] = (int8_t)rnd();
            for (int i = 0; i < N * K; i++) Bt[i] = (int8_t)rnd();
            const double ops = 2.0 * M * N * K;

            double t_ref, t_av = 0, t_x2 = 0;
            BENCH(t_ref, gemm_s8_ref(A, Bt, C, M, N, K));
            if (avx2) {
                BENCH(t_av, gemm_s8_autovec(A, Bt, C, M, N, K));
                BENCH(t_x2, gemm_s8_avx2(A, Bt, C, M, N, K));
            }
            char name[64];
            snprintf(name, sizeof name, "GEMM s8 %dx%dx%d", M, N, K);
            row(name, "ref", t_ref, ops, t_ref);
            if (avx2) {
                row(name, "autovec", t_av, ops, t_ref);
                row(name, "avx2", t_x2, ops, t_ref);
            }
            free(A); free(Bt); free(C);
        }
    }

    /* ---- requant (scalar baseline; SIMD version is a next step) -------- */
    {
        const size_t n = 1 << 16;
        int32_t *acc = malloc(sizeof(int32_t) * n);
        int8_t *out = malloc(n);
        for (size_t i = 0; i < n; i++) acc[i] = (int32_t)(rnd() & 0xfffff) - (1 << 19);
        double t;
        BENCH(t, requant_s32_to_s8_ref(acc, out, n, 1568669081, 9, -3));
        header("Gelem/s");
        row("requant s32->s8 n=65536", "ref", t, (double)n, t);
        free(acc); free(out);
    }

    /* ---- FFT (scalar baseline; the NEON/AVX2 version is the next rung) - */
    {
        header("GFLOP/s*");
        for (int n = 256; n <= 4096; n <<= 2) {
            fft_plan *p = fft_plan_create(n);
            float *src = malloc(sizeof(float) * 2 * (size_t)n);
            float *d = malloc(sizeof(float) * 2 * (size_t)n);
            for (int i = 0; i < 2 * n; i++) src[i] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
            int lg = 0; while ((1 << lg) < n) lg++;
            const double flops = 5.0 * n * lg; /* standard radix-2 FFT flop count */
            double t;
            BENCH(t, { memcpy(d, src, sizeof(float) * 2 * (size_t)n); fft_f32_ref(p, d); });
            char name[64];
            snprintf(name, sizeof name, "FFT c32 n=%d", n);
            row(name, "ref", t, flops, t);
            free(src); free(d); fft_plan_destroy(p);
        }
        printf("\n*FFT time includes a memcpy of the input. The AVX2 FFT steps are in ./build/bench_fft.\n");
    }
    return 0;
}
