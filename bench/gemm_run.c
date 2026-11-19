/*
 * Run ONE GEMM implementation many times, so perf stat / perf record see it alone.
 *   ./build/gemm_run <ref|autovec|s1|s2_name> <M> <N> <K> <iters>
 * e.g. taskset -c 2 perf stat -e cycles,instructions ./build/gemm_run s2 128 128 256 20000
 * The step is checked bit-exact against the reference first; a wrong one is refused.
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 4242u;
static uint32_t rnd(void) { rng = rng * 1664525u + 1013904223u; return rng >> 8; }

static gemm_fn find(const char *w, const char **full)
{
    if (!strcmp(w, "ref")) { *full = "ref"; return gemm_s8_ref; }
    if (!strcmp(w, "autovec")) { *full = "autovec"; return gemm_s8_autovec; }
    size_t len = strlen(w);
    for (int s = 0; s < gemm_avx2_num_steps(); s++) {
        const char *nm = gemm_avx2_step_name(s);
        if (!strcmp(nm, w) || (!strncmp(nm, w, len) && nm[len] == '_')) { *full = nm; return gemm_avx2_step_fn(s); }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s <ref|autovec|step> M N K iters\nsteps:", argv[0]);
        for (int s = 0; s < gemm_avx2_num_steps(); s++) fprintf(stderr, " %s", gemm_avx2_step_name(s));
        fprintf(stderr, "\n");
        return 2;
    }
    const int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    const long iters = atol(argv[5]);
    if (M < 1 || N < 1 || K < 1 || iters < 1) { fprintf(stderr, "bad arguments\n"); return 2; }
    const char *name = NULL;
    gemm_fn fn = find(argv[1], &name);
    if (!fn) { fprintf(stderr, "unknown step '%s'\n", argv[1]); return 2; }
    if (fn != gemm_s8_ref && !dsp_cpu_has_avx2()) { fprintf(stderr, "no AVX2 here\n"); return 2; }

    int8_t *A = malloc((size_t)M * K), *Bt = malloc((size_t)N * K);
    int32_t *C = malloc(sizeof(int32_t) * (size_t)M * N), *R = malloc(sizeof(int32_t) * (size_t)M * N);
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (int8_t)rnd();
    for (size_t i = 0; i < (size_t)N * K; i++) Bt[i] = (int8_t)rnd();
    gemm_s8_ref(A, Bt, R, M, N, K);
    fn(A, Bt, C, M, N, K);
    if (memcmp(C, R, sizeof(int32_t) * (size_t)M * N)) { fprintf(stderr, "%s gives WRONG results; not running it\n", name); return 1; }
    for (long i = 0; i < 20 && i < iters; i++) fn(A, Bt, C, M, N, K);

    int sink = 0;
    double t0 = now_ns();
    for (long i = 0; i < iters; i++) { fn(A, Bt, C, M, N, K); sink += C[i % ((long)M * N)]; }
    double dt = now_ns() - t0;
    printf("%s  %dx%dx%d  iters=%ld  total %.3f ms  %.2f us/GEMM  %.2f GOP/s  [checksum %d]\n", name, M, N, K, iters,
           dt / 1e6, dt / iters / 1e3, 2.0 * M * N * K * iters / dt, sink);
    free(A); free(Bt); free(C); free(R);
    return 0;
}
