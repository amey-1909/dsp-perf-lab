/* Step 5: cache blocking: the first 10 stages (len <= BLOCK) run block by block, so a block stays in L1 */
#include "dsp.h"
#include "fft_internal.h"
#include "fft_steps.h"
#include <immintrin.h>

/* Column m of the 8x8 DFT matrix, for the bit-reversed input order:
 *   X[k] = sum_m a[m] * W8^(rev3(m) * k),  k = 0..7   (one ymm holds all 8 k) */
static const float DFT_RE[8][8] __attribute__((aligned(32))) = {
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f},
    {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f},
    {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f},
    {1.0f, 0.70710678f, 0.0f, -0.70710678f, -1.0f, -0.70710678f, 0.0f, 0.70710678f},
    {1.0f, -0.70710678f, 0.0f, 0.70710678f, -1.0f, 0.70710678f, 0.0f, -0.70710678f},
    {1.0f, -0.70710678f, 0.0f, 0.70710678f, -1.0f, 0.70710678f, 0.0f, -0.70710678f},
    {1.0f, 0.70710678f, 0.0f, -0.70710678f, -1.0f, -0.70710678f, 0.0f, 0.70710678f},
};
static const float DFT_IM[8][8] __attribute__((aligned(32))) = {
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f},
    {0.0f, -0.70710678f, -1.0f, -0.70710678f, 0.0f, 0.70710678f, 1.0f, 0.70710678f},
    {0.0f, 0.70710678f, -1.0f, 0.70710678f, 0.0f, -0.70710678f, 1.0f, -0.70710678f},
    {0.0f, -0.70710678f, 1.0f, -0.70710678f, 0.0f, 0.70710678f, -1.0f, 0.70710678f},
    {0.0f, 0.70710678f, 1.0f, 0.70710678f, 0.0f, -0.70710678f, -1.0f, -0.70710678f},
};

static inline void deinterleave8(const float *p, __m256 *re, __m256 *im)
{
    __m256 x0 = _mm256_loadu_ps(p);
    __m256 x1 = _mm256_loadu_ps(p + 8);
    __m256 r = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 m = _mm256_shuffle_ps(x0, x1, _MM_SHUFFLE(3, 1, 3, 1));
    *re = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(r), _MM_SHUFFLE(3, 1, 2, 0)));
    *im = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(m), _MM_SHUFFLE(3, 1, 2, 0)));
}

static inline void interleave8(float *p, __m256 re, __m256 im)
{
    __m256 lo = _mm256_unpacklo_ps(re, im);
    __m256 hi = _mm256_unpackhi_ps(re, im);
    _mm256_storeu_ps(p,     _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(p + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

/* 8 complex values (16 floats) in place: 8-point DFT of the bit-reversed group */
static inline void dft8_direct(float *g)
{
    __m256 rr = _mm256_setzero_ps(), ii = _mm256_setzero_ps();
    __m256 ri = _mm256_setzero_ps(), ir = _mm256_setzero_ps();
    for (int m = 0; m < 8; m++) {
        __m256 ar = _mm256_broadcast_ss(g + 2 * m);
        __m256 ai = _mm256_broadcast_ss(g + 2 * m + 1);
        __m256 cr = _mm256_load_ps(DFT_RE[m]);
        __m256 ci = _mm256_load_ps(DFT_IM[m]);
        rr = _mm256_fmadd_ps(ar, cr, rr);
        ii = _mm256_fmadd_ps(ai, ci, ii);
        ri = _mm256_fmadd_ps(ar, ci, ri);
        ir = _mm256_fmadd_ps(ai, cr, ir);
    }
    interleave8(g, _mm256_sub_ps(rr, ii), _mm256_add_ps(ri, ir));
}

#define BLOCK 4096   /* complex values per block: 8 KB of data (+ ~8 KB of twiddles) fits in L1 */

/* One radix-2 stage (half >= 8) over `count` complex values starting at d.
 * j chunk outside, blocks inside: twiddles are loaded once per chunk. */
static inline __attribute__((always_inline)) vector_stage(const fft_plan *p, float *d, int count, int len)
{
    const int half = len >> 1;
    for (int j = 0; j < half; j += 8) {
        __m256 wr = _mm256_loadu_ps(p->tw_re + half + j);
        __m256 wi = _mm256_loadu_ps(p->tw_im + half + j);
        for (int i = 0; i < count; i += len) {
            __m256 b_0, b_1, a_0, a_1;
            deinterleave8(d + 2 * (i + j + half), &b_0, &b_1);
            deinterleave8(d + 2 * (i + j),        &a_0, &a_1);

            __m256 vr = _mm256_mul_ps(b_0, wr);
            vr = _mm256_fnmadd_ps(b_1, wi, vr);
            __m256 vi = _mm256_mul_ps(b_0, wi);
            vi = _mm256_fmadd_ps(b_1, wr, vi);

            interleave8(d + 2 * (i + j),        _mm256_add_ps(a_0, vr), _mm256_add_ps(a_1, vi));
            interleave8(d + 2 * (i + j + half), _mm256_sub_ps(a_0, vr), _mm256_sub_ps(a_1, vi));
        }
    }
}

/* Stages len = 2 .. m on one block of m complex values (8 <= m <= BLOCK, power of two) */
static inline __attribute__((always_inline)) fft_block(const fft_plan *p, float *blk, int m)
{
    for (int i = 0; i < m; i += 8)
        dft8_direct(blk + 2 * i);                  /* half = 1, 2, 4 */
    for (int len = 16; len <= m; len <<= 1)
        vector_stage(p, blk, m, len);              /* half = 8 .. m/2 */
}

void fft_f32_avx2_s5(const fft_plan *p, float *d)
{
    const int n = p->n;
     if (n <= BLOCK) { fft_f32_avx2_s4(p, d); return; }
    for (int i = 0; i < n; i++) {
        int j = p->rev[i];
        if (i < j) {
            float tr = d[2 * i], ti = d[2 * i + 1];
            d[2 * i] = d[2 * j];     d[2 * i + 1] = d[2 * j + 1];
            d[2 * j] = tr;           d[2 * j + 1] = ti;
        }
    }

    if (n < 8) {
        /* n = 2, 4: plain scalar stages */
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1;
            const int step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; j++) {
                    const float wr = p->tw[2 * j * step];
                    const float wi = p->tw[2 * j * step + 1];
                    float *a = d + 2 * (i + j);
                    float *b = d + 2 * (i + j + half);
                    const float vr = b[0] * wr - b[1] * wi;
                    const float vi = b[0] * wi + b[1] * wr;
                    const float ur = a[0], ui = a[1];
                    a[0] = ur + vr;  a[1] = ui + vi;
                    b[0] = ur - vr;  b[1] = ui - vi;
                }
            }
        }
        return;
    }
    const int m = n < BLOCK ? n : BLOCK;
    
    /* stages len = 2 .. m, one block at a time (each block stays in L1) */
    for (int b = 0; b < n; b += m)
        fft_block(p, d + 2 * b, m);

    /* remaining stages len = 2m .. n sweep the whole array */
    for (int len = 2 * m; len <= n; len <<= 1)
        vector_stage(p, d, n, len);
}