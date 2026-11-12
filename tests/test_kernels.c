/* Correctness tests: every variant vs an independent golden model.
 * Buffers are malloc'd at exact size so ASan / Valgrind catch overruns,
 * and shapes deliberately include awkward tails (n=1, K=19, N=7, ...). */
#include "dsp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rng_state = 88172645463325252ULL;
static uint32_t rnd(void)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 32);
}
static float rndf(void) { return (float)rnd() / 2147483648.0f - 1.0f; } /* [-1,1) */
static int8_t rnd8(void) { return (int8_t)(rnd() & 0xff); }              /* incl. -128 */

static int g_checks, g_fail;
#define CHECK(cond, ...)                                              \
    do {                                                              \
        g_checks++;                                                   \
        if (!(cond)) {                                                \
            g_fail++;                                                 \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);               \
            printf(__VA_ARGS__);                                      \
            printf("\n");                                             \
        }                                                             \
    } while (0)

/* ------------------------------------------------------------------ FIR */
static void test_fir(int n, int taps, int has_avx2)
{
    float *x = malloc(sizeof(float) * (size_t)(n + taps - 1));
    float *h = malloc(sizeof(float) * (size_t)taps);
    float *y_ref = malloc(sizeof(float) * (size_t)n);
    float *y_av = malloc(sizeof(float) * (size_t)n);
    float *y_x2 = malloc(sizeof(float) * (size_t)n);
    double *gold = malloc(sizeof(double) * (size_t)n);

    for (int i = 0; i < n + taps - 1; i++) x[i] = rndf();
    for (int k = 0; k < taps; k++) h[k] = rndf();
    for (int i = 0; i < n; i++) {
        double s = 0;
        for (int k = 0; k < taps; k++) s += (double)h[k] * (double)x[i + k];
        gold[i] = s;
    }

    fir_f32_ref(x, h, y_ref, n, taps);
    if (has_avx2) fir_f32_autovec(x, h, y_av, n, taps);
    if (has_avx2) fir_f32_avx2(x, h, y_x2, n, taps);

    double e_ref = 0, e_av = 0, e_x2 = 0;
    for (int i = 0; i < n; i++) {
        e_ref = fmax(e_ref, fabs(y_ref[i] - gold[i]));
        if (has_avx2) {
            e_av = fmax(e_av, fabs(y_av[i] - gold[i]));
            e_x2 = fmax(e_x2, fabs(y_x2[i] - gold[i]));
        }
    }
    const double tol = 1e-4;
    CHECK(e_ref < tol, "fir ref n=%d taps=%d err=%g", n, taps, e_ref);
    if (has_avx2) {
        CHECK(e_av < tol, "fir autovec n=%d taps=%d err=%g", n, taps, e_av);
        CHECK(e_x2 < tol, "fir avx2 n=%d taps=%d err=%g", n, taps, e_x2);
    }
    free(x); free(h); free(y_ref); free(y_av); free(y_x2); free(gold);
}

/* ----------------------------------------------------------------- GEMM */
static void test_gemm(int M, int N, int K, int has_avx2)
{
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)K * N);
    int8_t *Bt = malloc((size_t)N * K);
    int32_t *C0 = malloc(sizeof(int32_t) * (size_t)M * N);
    int32_t *C1 = malloc(sizeof(int32_t) * (size_t)M * N);
    int32_t *C2 = malloc(sizeof(int32_t) * (size_t)M * N);

    for (int i = 0; i < M * K; i++) A[i] = rnd8();
    for (int i = 0; i < K * N; i++) B[i] = rnd8();
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++) Bt[(size_t)j * K + k] = B[(size_t)k * N + j];

    gemm_s8_ref(A, Bt, C0, M, N, K);
    if (has_avx2) {
        gemm_s8_autovec(A, Bt, C1, M, N, K);
        gemm_s8_avx2(A, Bt, C2, M, N, K);
    }

    /* Golden uses the ORIGINAL (non-transposed) B, so it also validates the
     * Bt convention, not just agreement between our own kernels. */
    int bad_ref = 0, bad_av = 0, bad_x2 = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            int64_t s = 0;
            for (int k = 0; k < K; k++)
                s += (int64_t)A[(size_t)i * K + k] * (int64_t)B[(size_t)k * N + j];
            size_t idx = (size_t)i * N + j;
            bad_ref += (C0[idx] != (int32_t)s);
            if (has_avx2) {
                bad_av += (C1[idx] != (int32_t)s);
                bad_x2 += (C2[idx] != (int32_t)s);
            }
        }
    CHECK(bad_ref == 0, "gemm ref %dx%dx%d mismatches=%d", M, N, K, bad_ref);
    if (has_avx2) {
        CHECK(bad_av == 0, "gemm autovec %dx%dx%d mismatches=%d", M, N, K, bad_av);
        CHECK(bad_x2 == 0, "gemm avx2 %dx%dx%d mismatches=%d", M, N, K, bad_x2);
    }
    free(A); free(B); free(Bt); free(C0); free(C1); free(C2);
}

/* -------------------------------------------------------------- requant */
static void test_requant(int shift, double real_mult_q31)
{
    const size_t n = 4096;
    const int32_t zp = -3;
    int32_t mult = (int32_t)(real_mult_q31 * 2147483648.0);
    int32_t *acc = malloc(sizeof(int32_t) * n);
    int8_t *out = malloc(n);
    for (size_t i = 0; i < n; i++) acc[i] = (int32_t)(rnd() >> 12) - (1 << 19); /* +-2^19 */
    requant_s32_to_s8_ref(acc, out, n, mult, shift, zp);

    int worst = 0;
    double scale = ((double)mult / 2147483648.0) / (double)(1 << shift);
    for (size_t i = 0; i < n; i++) {
        double e = round((double)acc[i] * scale) + zp;
        if (e > 127) e = 127;
        if (e < -128) e = -128;
        int d = abs((int)out[i] - (int)e);
        if (d > worst) worst = d;
    }
    CHECK(worst <= 1, "requant shift=%d worst diff=%d", shift, worst);
    free(acc); free(out);
}

/* ------------------------------------------------------------------ FFT */
static double rel_l2_err(const float *x, const double *g, int n2)
{
    double num = 0, den = 0;
    for (int i = 0; i < n2; i++) {
        num += ((double)x[i] - g[i]) * ((double)x[i] - g[i]);
        den += g[i] * g[i];
    }
    return sqrt(num / den);
}

static void test_fft(int n, int has_avx2)
{
    fft_plan *p = fft_plan_create(n);
    CHECK(p != NULL, "fft plan n=%d", n);
    if (!p) return;
    const size_t bytes = sizeof(float) * 2 * (size_t)n;
    float *src = malloc(bytes);
    float *out = malloc(bytes);
    double *g = malloc(sizeof(double) * 2 * (size_t)n);
    for (int i = 0; i < 2 * n; i++) src[i] = rndf();

    for (int k = 0; k < n; k++) { /* naive DFT in double */
        double re = 0, im = 0;
        for (int t = 0; t < n; t++) {
            double a = -2.0 * 3.14159265358979323846 * (double)((long)k * t % n) / n;
            re += src[2 * t] * cos(a) - src[2 * t + 1] * sin(a);
            im += src[2 * t] * sin(a) + src[2 * t + 1] * cos(a);
        }
        g[2 * k] = re; g[2 * k + 1] = im;
    }

    memcpy(out, src, bytes);
    fft_f32_ref(p, out);
    double rel = rel_l2_err(out, g, 2 * n);
    CHECK(rel < 1e-5, "fft ref n=%d rel L2 err=%g", n, rel);

    if (has_avx2) {
        /* every registered AVX2 step must be correct on its own */
        for (int s = 0; s < fft_avx2_num_steps(); s++) {
            memcpy(out, src, bytes);
            fft_avx2_step_fn(s)(p, out);
            rel = rel_l2_err(out, g, 2 * n);
            CHECK(rel < 1e-5, "fft %s n=%d rel L2 err=%g", fft_avx2_step_name(s), n, rel);
        }
    }
    CHECK(fft_plan_create(100) == NULL, "fft rejects non power-of-two");
    free(src); free(out); free(g); fft_plan_destroy(p);
}

/* ------------------------------------------------------------------ SAD */
static uint32_t sad_gold(const uint8_t *cur, int cs, const uint8_t *ref, int rs)
{
    int64_t s = 0;
    for (int y = 0; y < 16; y++)
        for (int x = 0; x < 16; x++) {
            int64_t d = (int64_t)cur[y * cs + x] - (int64_t)ref[y * rs + x];
            s += d < 0 ? -d : d;
        }
    return (uint32_t)s;
}

static void test_sad(int has_avx2)
{
    enum { W = 80, H = 72 };
    /* Exact-size buffers so ASan/Valgrind catch any read past the block. */
    uint8_t *ref = malloc((size_t)W * H);
    uint8_t *curf = malloc((size_t)W * H);   /* cur block lives inside a bigger frame */
    uint8_t *curc = malloc(16 * 16);         /* ...and also as a tight 16x16 block */

    for (int trial = 0; trial < 300; trial++) {
        for (int i = 0; i < W * H; i++) { ref[i] = (uint8_t)rnd(); curf[i] = (uint8_t)rnd(); }
        if (trial == 0) { memset(ref, 0, (size_t)W * H); memset(curf, 255, (size_t)W * H); } /* max SAD */
        if (trial == 1) { memset(ref, 255, (size_t)W * H); memset(curf, 0, (size_t)W * H); }
        const int tight = trial & 1;
        int cx = (int)(rnd() % (W - 15)), cy = (int)(rnd() % (H - 15));
        const uint8_t *cur = tight ? curc : curf + (size_t)cy * W + cx;
        const int cs = tight ? 16 : W;
        if (tight)
            for (int y = 0; y < 16; y++) memcpy(curc + y * 16, curf + (size_t)(cy + y) * W + cx, 16);

        const uint8_t *r[4];
        uint32_t gold[4];
        for (int c = 0; c < 4; c++) {
            int x = (int)(rnd() % (W - 15)), y = (int)(rnd() % (H - 15));
            r[c] = ref + (size_t)y * W + x;
            gold[c] = sad_gold(cur, cs, r[c], W);
        }

        uint32_t o_ref[4], o_av[4], o_x2[4];
        sad_x4_16x16_ref(cur, cs, r[0], r[1], r[2], r[3], W, o_ref);
        if (has_avx2) {
            sad_x4_16x16_autovec(cur, cs, r[0], r[1], r[2], r[3], W, o_av);
            sad_x4_16x16_avx2(cur, cs, r[0], r[1], r[2], r[3], W, o_x2);
        }
        for (int c = 0; c < 4; c++) {
            CHECK(sad_16x16_ref(cur, cs, r[c], W) == gold[c], "sad ref trial=%d c=%d", trial, c);
            CHECK(o_ref[c] == gold[c], "sad_x4 ref trial=%d c=%d", trial, c);
            if (has_avx2) {
                CHECK(sad_16x16_autovec(cur, cs, r[c], W) == gold[c], "sad autovec trial=%d c=%d", trial, c);
                CHECK(sad_16x16_avx2(cur, cs, r[c], W) == gold[c], "sad avx2 trial=%d c=%d", trial, c);
                CHECK(o_av[c] == gold[c], "sad_x4 autovec trial=%d c=%d", trial, c);
                CHECK(o_x2[c] == gold[c], "sad_x4 avx2 trial=%d c=%d", trial, c);
            }
        }
    }
    free(ref); free(curf); free(curc);
}

int main(void)
{
    const int has_avx2 = dsp_cpu_has_avx2();
    printf("AVX2+FMA available: %s\n", has_avx2 ? "yes" : "no (avx2 variants skipped)");

    const int fir_n[] = {1, 7, 8, 31, 32, 33, 1000, 4097};
    const int fir_t[] = {1, 3, 16, 33, 64};
    for (size_t a = 0; a < sizeof fir_n / sizeof *fir_n; a++)
        for (size_t b = 0; b < sizeof fir_t / sizeof *fir_t; b++)
            test_fir(fir_n[a], fir_t[b], has_avx2);

    const int g[][3] = {{1, 1, 1}, {5, 7, 19}, {17, 13, 33}, {4, 4, 16},
                        {64, 64, 64}, {33, 50, 100}, {8, 9, 4095}};
    for (size_t i = 0; i < sizeof g / sizeof *g; i++)
        test_gemm(g[i][0], g[i][1], g[i][2], has_avx2);

    test_requant(9, 0.73);
    test_requant(0, 0.5);
    test_requant(14, 0.99);

    for (int n = 2; n <= 16384; n <<= 1) test_fft(n, has_avx2);

    test_sad(has_avx2);

    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
