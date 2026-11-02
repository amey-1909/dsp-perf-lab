/*
 * Run ONE kernel variant many times, so `perf stat` measures it alone.
 *
 *   ./build/kernel_run <kernel> <variant> <iters>
 *
 *   kernel  : fir | gemm | sad | sadx4
 *   variant : ref | autovec | avx2
 *   iters   : how many times to run the kernel body
 *
 * What one iteration is (and the "unit" used to normalise results):
 *   fir   : FIR over 4096 outputs, 64 taps, float          unit = 1 output sample
 *   gemm  : int8 GEMM 128 x 128 x 256 (B pre-transposed)   unit = 1 multiply-accumulate
 *   sad   : four single 16x16 SADs (candidates 1 px apart) unit = 1 SAD
 *   sadx4 : one sad_x4 call (same four candidates)         unit = 1 SAD
 *
 * All data sits in L1/L2 so the numbers reflect the core, not memory. Before
 * timing, the variant's output is checked against the scalar reference and a
 * wrong variant is refused. The last line printed is machine-readable (RESULT ...)
 * and is used by scripts/perf_kernels.sh.
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

typedef void (*fir_fn)(const float *, const float *, float *, int, int);
typedef void (*gemm_fn)(const int8_t *, const int8_t *, int32_t *, int, int, int);
typedef uint32_t (*sad_fn)(const uint8_t *, int, const uint8_t *, int);
typedef void (*sadx4_fn)(const uint8_t *, int, const uint8_t *, const uint8_t *,
                         const uint8_t *, const uint8_t *, int, uint32_t *);

static volatile double g_sink;

static int bad(const char *what)
{
    fprintf(stderr, "%s gives WRONG results; not running it\n", what);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <fir|gemm|sad|sadx4> <ref|autovec|avx2> <iters>\n", argv[0]);
        return 2;
    }
    const char *kernel = argv[1], *variant = argv[2];
    const long iters = atol(argv[3]);
    const int v = !strcmp(variant, "ref") ? 0 : !strcmp(variant, "autovec") ? 1 : !strcmp(variant, "avx2") ? 2 : -1;
    if (v < 0 || iters < 1) {
        fprintf(stderr, "variant must be ref|autovec|avx2 and iters >= 1\n");
        return 2;
    }
    if (v > 0 && !dsp_cpu_has_avx2()) {
        fprintf(stderr, "no AVX2+FMA on this machine\n");
        return 2;
    }

    double t0 = 0, dt = 0;
    double units_per_iter = 0;
    const char *unit = "";

    if (!strcmp(kernel, "fir")) {
        enum { N = 4096, TAPS = 64 };
        static float x[N + TAPS], h[TAPS], y[N], yr[N];
        for (int i = 0; i < N + TAPS; i++) x[i] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        for (int k = 0; k < TAPS; k++) h[k] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        fir_fn fn = v == 0 ? fir_f32_ref : v == 1 ? fir_f32_autovec : fir_f32_avx2;
        fir_f32_ref(x, h, yr, N, TAPS);
        fn(x, h, y, N, TAPS);
        for (int i = 0; i < N; i++) if (fabsf(y[i] - yr[i]) > 1e-4f) return bad("fir");
        for (long i = 0; i < 20 && i < iters; i++) fn(x, h, y, N, TAPS);
        t0 = now_ns();
        for (long i = 0; i < iters; i++) fn(x, h, y, N, TAPS);
        dt = now_ns() - t0;
        g_sink = y[1];
        units_per_iter = N; unit = "output";
    } else if (!strcmp(kernel, "gemm")) {
        enum { M = 128, Nn = 128, K = 256 };
        static int8_t A[M * K], Bt[Nn * K];
        static int32_t C[M * Nn], Cr[M * Nn];
        for (int i = 0; i < M * K; i++) A[i] = (int8_t)rnd();
        for (int i = 0; i < Nn * K; i++) Bt[i] = (int8_t)rnd();
        gemm_fn fn = v == 0 ? gemm_s8_ref : v == 1 ? gemm_s8_autovec : gemm_s8_avx2;
        gemm_s8_ref(A, Bt, Cr, M, Nn, K);
        fn(A, Bt, C, M, Nn, K);
        if (memcmp(C, Cr, sizeof C) != 0) return bad("gemm");
        for (long i = 0; i < 3 && i < iters; i++) fn(A, Bt, C, M, Nn, K);
        t0 = now_ns();
        for (long i = 0; i < iters; i++) fn(A, Bt, C, M, Nn, K);
        dt = now_ns() - t0;
        g_sink = C[1];
        units_per_iter = (double)M * Nn * K; unit = "MAC";
    } else if (!strcmp(kernel, "sad") || !strcmp(kernel, "sadx4")) {
        enum { S = 64 };
        static uint8_t frame[S * S], cur[16 * 16];
        for (int i = 0; i < S * S; i++) frame[i] = (uint8_t)rnd();
        for (int i = 0; i < 16 * 16; i++) cur[i] = (uint8_t)rnd();
        const uint8_t *b = frame + 8 * S + 8;   /* 4 candidates, 1 pixel apart */
        uint32_t gold[4], out[4];
        for (int c = 0; c < 4; c++) gold[c] = sad_16x16_ref(cur, 16, b + c, S);
        const int x4 = !strcmp(kernel, "sadx4");
        sad_fn f1 = v == 0 ? sad_16x16_ref : v == 1 ? sad_16x16_autovec : sad_16x16_avx2;
        sadx4_fn f4 = v == 0 ? sad_x4_16x16_ref : v == 1 ? sad_x4_16x16_autovec : sad_x4_16x16_avx2;
        if (x4) {
            f4(cur, 16, b, b + 1, b + 2, b + 3, S, out);
            if (memcmp(out, gold, sizeof gold) != 0) return bad("sadx4");
        } else {
            for (int c = 0; c < 4; c++) if (f1(cur, 16, b + c, S) != gold[c]) return bad("sad");
        }
        uint32_t acc = 0;
        for (long i = 0; i < 1000 && i < iters; i++) { f4(cur, 16, b, b + 1, b + 2, b + 3, S, out); acc += out[0]; }
        t0 = now_ns();
        if (x4) {
            for (long i = 0; i < iters; i++) { f4(cur, 16, b, b + 1, b + 2, b + 3, S, out); acc += out[0] + out[3]; }
        } else {
            for (long i = 0; i < iters; i++)
                for (int c = 0; c < 4; c++) acc += f1(cur, 16, b + c, S);
        }
        dt = now_ns() - t0;
        g_sink = acc;
        units_per_iter = 4; unit = "SAD";
    } else {
        fprintf(stderr, "unknown kernel '%s'\n", kernel);
        return 2;
    }

    const double ns_unit = dt / ((double)iters * units_per_iter);
    printf("%s %s: %ld iters, %.3f ms, %.4f ns per %s\n", kernel, variant, iters, dt / 1e6, ns_unit, unit);
    printf("RESULT kernel=%s variant=%s iters=%ld units_per_iter=%.0f unit=%s ns_per_unit=%.6f ns_per_iter=%.3f\n",
           kernel, variant, iters, units_per_iter, unit, ns_unit, dt / (double)iters);
    return 0;
}
