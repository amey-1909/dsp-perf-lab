/*
 * Compare the best AVX2 step of this repo with FFTW (single precision).
 *
 *   ./build/bench_fftw              # sizes 256 1024 4096 16384 65536
 *   ./build/bench_fftw 1024 8192    # chosen sizes (powers of two)
 *
 * Built only if CMake finds fftw3.h and libfftw3f (sudo apt install libfftw3-dev).
 *
 * Same conditions for every row: single-precision complex-to-complex, in place,
 * forward transform (exp(-j...), the convention of this repo), 32-byte aligned
 * buffer, one memcpy of the input per FFT to re-arm the in-place transform, all
 * functions timed in interleaved rounds with each keeping its best round.
 * FFTW is shown planned with FFTW_MEASURE (normal usage) and FFTW_ESTIMATE.
 * Planning time is not included. Every implementation is checked against the
 * scalar reference first; a wrong one is reported and not timed.
 *
 * "x FFTW" is time / time(FFTW_MEASURE): 2.00x means that row takes twice as long.
 * Pin the process:  taskset -c 2 ./build/bench_fftw
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUNDS 25
#define NF 4

static uint32_t rng = 12345u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static double rel_l2(const float *a, const float *ref, int n2)
{
    double num = 0, den = 0;
    for (int i = 0; i < n2; i++) {
        double d = (double)a[i] - (double)ref[i];
        num += d * d;
        den += (double)ref[i] * (double)ref[i];
    }
    return sqrt(num / den);
}

typedef struct {
    const char *name;
    int kind;               /* 0 = repo function, 1 = FFTW plan */
    fft_fn fn;
    fftwf_plan plan;
} impl;

static void run_one(const impl *m, const fft_plan *p, float *buf)
{
    if (m->kind == 0) m->fn(p, buf);
    else fftwf_execute_dft(m->plan, (fftwf_complex *)buf, (fftwf_complex *)buf);
}

static double time_batch(const impl *m, const fft_plan *p, float *buf, const float *src,
                         size_t bytes, long iters)
{
    const double t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        memcpy(buf, src, bytes);
        run_one(m, p, buf);
    }
    return (now_ns() - t0) / (double)iters;
}

int main(int argc, char **argv)
{
    int sizes[16] = {256, 1024, 4096, 16384, 65536};
    int nsizes = 5;
    if (argc > 1) {
        nsizes = 0;
        for (int a = 1; a < argc && nsizes < 16; a++) {
            int n = atoi(argv[a]);
            if (n < 2 || (n & (n - 1))) {
                fprintf(stderr, "size %s is not a power of two >= 2\n", argv[a]);
                return 2;
            }
            sizes[nsizes++] = n;
        }
    }
    const int nsteps = fft_avx2_num_steps();
    if (nsteps < 1) { fprintf(stderr, "no AVX2 steps registered\n"); return 2; }
    const int best_step = nsteps - 1;

    printf("# Best AVX2 step (%s) vs FFTW (single precision)\n\n", fft_avx2_step_name(best_step));
    printf("| %6s | %-22s | %10s | %8s | %8s | %9s | %-12s |\n", "n", "implementation",
           "time (us)", "GFLOP/s*", "vs ref", "x FFTW", "check");
    printf("|%s|%s|%s|%s|%s|%s|%s|\n", "--------:", "------------------------", "-----------:",
           "---------:", "---------:", "----------:", "--------------");

    for (int si = 0; si < nsizes; si++) {
        const int n = sizes[si];
        const size_t bytes = sizeof(float) * 2 * (size_t)n;
        fft_plan *p = fft_plan_create(n);
        float *buf = (float *)fftwf_malloc(bytes);          /* 32-byte aligned for all rows */
        float *src = malloc(bytes), *ref_out = malloc(bytes);
        if (!p || !buf || !src || !ref_out) { fprintf(stderr, "alloc failed\n"); return 1; }

        /* plan first: FFTW_MEASURE overwrites the buffer while planning */
        fftwf_plan pm = fftwf_plan_dft_1d(n, (fftwf_complex *)buf, (fftwf_complex *)buf,
                                          FFTW_FORWARD, FFTW_MEASURE);
        fftwf_plan pe = fftwf_plan_dft_1d(n, (fftwf_complex *)buf, (fftwf_complex *)buf,
                                          FFTW_FORWARD, FFTW_ESTIMATE);
        if (!pm || !pe) { fprintf(stderr, "FFTW planning failed for n=%d\n", n); return 1; }
         {                                                      /* <-- new block starts here */
            double add, mul, fma;
            fftwf_flops(pm, &add, &mul, &fma);
            int lg0 = 0;
            while ((1 << lg0) < n) lg0++;
            printf("<!-- FFTW n=%d: add=%.0f mul=%.0f fma=%.0f -> %.0f flops (radix-2 count 5*n*log2n = %.0f) -->\n",
                   n, add, mul, fma, add + mul + 2 * fma, 5.0 * n * lg0);
            fftwf_print_plan(pm);
            printf("\n");
        }                                                      /* <-- new block ends here */

        for (int i = 0; i < 2 * n; i++) src[i] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        int lg = 0;
        while ((1 << lg) < n) lg++;
        const double flops = 5.0 * n * lg;

        memcpy(ref_out, src, bytes);
        fft_f32_ref(p, ref_out);

        char best_name[48];
        snprintf(best_name, sizeof best_name, "ours %s", fft_avx2_step_name(best_step));
        impl m[NF] = {
            {"ref (scalar)",       0, fft_f32_ref,                 NULL},
            {best_name,            0, fft_avx2_step_fn(best_step), NULL},
            {"FFTW_MEASURE",       1, NULL,                        pm},
            {"FFTW_ESTIMATE",      1, NULL,                        pe},
        };
        int ok[NF] = {0};
        double err[NF] = {0}, best[NF], t_iters[NF];
        long iters[NF] = {0};
        for (int f = 0; f < NF; f++) {
            memcpy(buf, src, bytes);
            run_one(&m[f], p, buf);
            err[f] = rel_l2(buf, ref_out, 2 * n);
            ok[f] = err[f] <= 1e-4;
            best[f] = 1e30;
            if (!ok[f]) continue;
            long it = 1;
            for (;;) {                                /* calibrate: batch of ~5 ms */
                t_iters[f] = time_batch(&m[f], p, buf, src, bytes, it);
                if (t_iters[f] * (double)it > 5e6) break;
                it *= 2;
            }
            iters[f] = it;
        }
        for (int r = 0; r < ROUNDS; r++)
            for (int k = 0; k < NF; k++) {
                const int f = (k + r) % NF;
                if (!ok[f]) continue;
                const double t = time_batch(&m[f], p, buf, src, bytes, iters[f]);
                if (t < best[f]) best[f] = t;
            }

        for (int f = 0; f < NF; f++) {
            if (!ok[f]) {
                char msg[32];
                snprintf(msg, sizeof msg, "WRONG %.0e", err[f]);
                printf("| %6d | %-22s | %10s | %8s | %8s | %9s | %-12s |\n", n, m[f].name, "-",
                       "-", "-", "-", msg);
                continue;
            }
            char chk[32];
            snprintf(chk, sizeof chk, "ok (%.1e)", err[f]);
            printf("| %6d | %-22s | %10.2f | %8.2f | %7.2fx | %8.2fx | %-12s |\n", n, m[f].name,
                   best[f] / 1000.0, flops / best[f], best[0] / best[f],
                   ok[2] ? best[f] / best[2] : 0.0, chk);
        }
        fftwf_destroy_plan(pm);
        fftwf_destroy_plan(pe);
        fftwf_free(buf);
        free(src); free(ref_out);
        fft_plan_destroy(p);
    }
    printf("\n*5 n log2(n) flops / time. 'x FFTW' = time / time(FFTW_MEASURE); 'check' = relative L2 error vs the scalar reference.\n");
    return 0;
}
