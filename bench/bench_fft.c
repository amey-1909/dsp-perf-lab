/*
 * FFT step-ladder benchmark.
 *
 * For every FFT size, prints the scalar reference and then each AVX2 step in
 * order, with two speedups: vs the scalar ref, and vs the PREVIOUS step. The
 * second column is the point of this tool: it says exactly what each change
 * bought. Every step's output is checked against the reference first, and a
 * step that gives wrong results is reported and not timed.
 *
 *   ./build/bench_fft              # sizes 256 1024 4096 16384
 *   ./build/bench_fft 1024 8192    # chosen sizes (powers of two)
 *
 * Method: all functions are timed in INTERLEAVED rounds (ref, step1, step2,
 * ..., then again, with the starting function rotated each round) and each
 * function keeps its best round. Timing one function for a long stretch and
 * then the next lets slow drift (frequency changes, a noisy neighbour on a
 * shared machine) bias whichever function ran during the bad stretch;
 * interleaving spreads that over all of them. Even so, differences of a few
 * percent are within noise: run it several times, and treat identical code
 * showing anything other than ~1.00x as the noise floor of your machine.
 *
 * Pin the process for stable numbers:  taskset -c 2 ./build/bench_fft
 * Timing includes one memcpy of the input per FFT (re-arms the in-place
 * transform); it is identical for every row.
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUNDS 25
#define MAX_FUNCS 64

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

/* Average ns per FFT over `iters` back-to-back transforms. */
static double time_batch(fft_fn fn, const fft_plan *p, float *d, const float *src,
                         size_t bytes, long iters)
{
    const double t0 = now_ns();
    for (long i = 0; i < iters; i++) {
        memcpy(d, src, bytes);
        fn(p, d);
    }
    return (now_ns() - t0) / (double)iters;
}

static void row(int n, const char *name, double ns, double flops, double t_ref,
                double t_prev, const char *check)
{
    printf("| %6d | %-16s | %10.2f | %8.2f | %7.2fx | %8.2fx | %-9s |\n", n, name,
           ns / 1000.0, flops / ns, t_ref / ns, t_prev / ns, check);
}

int main(int argc, char **argv)
{
    if (!dsp_cpu_has_avx2()) {
        printf("This machine has no AVX2+FMA; bench_fft needs it.\n");
        return 0;
    }

    int sizes[16] = {256, 1024, 4096, 16384};
    int nsizes = 4;
    if (argc > 1) {
        nsizes = 0;
        for (int a = 1; a < argc && nsizes < 16; a++) {
            int n = atoi(argv[a]);
            if (n < 2 || (n & (n - 1)) != 0) {
                fprintf(stderr, "size %s is not a power of two >= 2\n", argv[a]);
                return 1;
            }
            sizes[nsizes++] = n;
        }
    }

    const int nsteps = fft_avx2_num_steps();
    if (nsteps + 1 > MAX_FUNCS) {
        fprintf(stderr, "too many steps (max %d)\n", MAX_FUNCS - 1);
        return 1;
    }
    printf("# FFT step ladder (%d AVX2 step%s)\n\n", nsteps, nsteps == 1 ? "" : "s");
    for (int s = 0; s < nsteps; s++)
        printf("- %s: %s\n", fft_avx2_step_name(s), fft_avx2_step_desc(s));
    printf("\n| %6s | %-16s | %10s | %8s | %8s | %9s | %-9s |\n", "n", "step", "time (us)",
           "GFLOP/s*", "vs ref", "vs prev", "check");
    printf("|%s|%s|%s|%s|%s|%s|%s|\n", "--------:", "------------------", "-----------:",
           "---------:", "---------:", "----------:", "-----------");

    for (int si = 0; si < nsizes; si++) {
        const int n = sizes[si];
        const size_t bytes = sizeof(float) * 2 * (size_t)n;
        fft_plan *p = fft_plan_create(n);
        float *src = malloc(bytes), *d = malloc(bytes), *ref_out = malloc(bytes);
        for (int i = 0; i < 2 * n; i++) src[i] = (float)(rnd() & 0xffff) / 32768.0f - 1.0f;
        int lg = 0;
        while ((1 << lg) < n) lg++;
        const double flops = 5.0 * n * lg;   /* standard radix-2 FFT flop count */

        memcpy(ref_out, src, bytes);
        fft_f32_ref(p, ref_out);

        /* function 0 = scalar ref, 1..nsteps = AVX2 steps */
        fft_fn fn[MAX_FUNCS] = {0};
        int ok[MAX_FUNCS] = {0};
        double err[MAX_FUNCS] = {0};
        long iters[MAX_FUNCS] = {0};
        double best[MAX_FUNCS] = {0};
        const int nf = nsteps + 1;
        fn[0] = fft_f32_ref; ok[0] = 1; err[0] = 0;
        for (int s = 0; s < nsteps; s++) {
            fn[s + 1] = fft_avx2_step_fn(s);
            memcpy(d, src, bytes);
            fn[s + 1](p, d);
            err[s + 1] = rel_l2(d, ref_out, 2 * n);
            ok[s + 1] = err[s + 1] <= 1e-4;   /* never time a broken kernel */
        }

        for (int f = 0; f < nf; f++) {        /* calibrate: batch of ~5 ms */
            best[f] = 1e30;
            if (!ok[f]) continue;
            long it = 1;
            for (;;) {
                if (time_batch(fn[f], p, d, src, bytes, it) * (double)it > 5e6) break;
                it *= 2;
            }
            iters[f] = it;
        }
        for (int r = 0; r < ROUNDS; r++) {    /* interleaved rounds, rotating start */
            for (int k = 0; k < nf; k++) {
                const int f = (k + r) % nf;
                if (!ok[f]) continue;
                const double t = time_batch(fn[f], p, d, src, bytes, iters[f]);
                if (t < best[f]) best[f] = t;
            }
        }

        const double t_ref = best[0];
        row(n, "ref (scalar)", t_ref, flops, t_ref, t_ref, "-");
        double t_prev = t_ref;
        for (int s = 0; s < nsteps; s++) {
            const int f = s + 1;
            if (!ok[f]) {
                char msg[32];
                snprintf(msg, sizeof msg, "WRONG %.0e", err[f]);
                printf("| %6d | %-16s | %10s | %8s | %8s | %9s | %-9s |\n", n,
                       fft_avx2_step_name(s), "-", "-", "-", "-", msg);
                continue;
            }
            row(n, fft_avx2_step_name(s), best[f], flops, t_ref, t_prev, "ok");
            t_prev = best[f];
        }
        free(src); free(d); free(ref_out);
        fft_plan_destroy(p);
    }
    printf("\n*5 n log2(n) flops / time. 'vs prev' compares with the previous step that passed its check.\n");
    return 0;
}
