/* Step 4: column blocking: widen a chunk of Bt to int16 into a small L2-resident buffer, reuse it for all rows */
#include "dsp.h"
#include "gemm_steps.h"
#include <immintrin.h>
#include <stdlib.h>

/* int16 chunk of Bt kept resident in L2 (256 KB on the test machine) next to A rows and C */
#define CHUNK_BYTES (128 * 1024)

static inline __m256i ld16(const int8_t *p)
{
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p));
}

/* rows of W are padded to a multiple of 16 int16 and 32-byte aligned: aligned loads */
static inline __m256i ld16w(const int16_t *p)
{
    return _mm256_load_si256((const __m256i *)p);
}

static inline int32_t hsum(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static int32_t dot1(const int8_t *a, const int16_t *b, int K)
{
    const int K16 = K & ~15;
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < K16; k += 16) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ld16(a + k), ld16w(b + k)));
    int32_t s = hsum(acc);
    for (int k = K16; k < K; k++) s += (int32_t)a[k] * (int32_t)b[k];
    return s;
}

void gemm_s8_avx2_s4(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    const int M2 = M & ~1, K16 = K & ~15;
    const int Kp = (K + 15) & ~15;                       /* padded row stride of W */

    int NB = CHUNK_BYTES / (Kp * (int)sizeof(int16_t));  /* columns per chunk */
    NB &= ~3;
    if (NB < 4) NB = 4;
    if (NB > N) NB = N;

    static int16_t *W_buf;
    static size_t W_cap;
    const size_t need = (size_t)NB * Kp;
    if (need > W_cap) {
        free(W_buf);
        W_cap = need;
        W_buf = (int16_t *)aligned_alloc(32, ((need * sizeof(int16_t) + 31) / 32) * 32 + 32);
        if (!W_buf) { W_cap = 0; return; }
    }
    int16_t *W = W_buf;

    for (int jb = 0; jb < N; jb += NB) {
        const int nb = (N - jb < NB) ? N - jb : NB;
        const int nb4 = nb & ~3;

        /* widen this chunk of Bt once; it then stays in L2 while all rows use it */
        for (int j = 0; j < nb; j++) {
            const int8_t *src = Bt + (size_t)(jb + j) * K;
            int16_t *dst = W + (size_t)j * Kp;
            int k = 0;
            for (; k + 16 <= K; k += 16) _mm256_store_si256((__m256i *)(dst + k), ld16(src + k));
            for (; k < Kp; k++) dst[k] = k < K ? src[k] : 0;
        }

        for (int i = 0; i < M2; i += 2) {
            const int8_t *a0 = A + (size_t)i * K, *a1 = a0 + K;
            int j = 0;
            for (; j < nb4; j += 4) {
                const int16_t *b0 = W + (size_t)j * Kp, *b1 = b0 + Kp, *b2 = b1 + Kp, *b3 = b2 + Kp;
                __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
                __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
                for (int k = 0; k < K16; k += 16) {
                    const __m256i x0 = ld16(a0 + k), x1 = ld16(a1 + k);
                    __m256i y;
                    y = ld16w(b0 + k); c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x0, y)); c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x1, y));
                    y = ld16w(b1 + k); c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x0, y)); c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x1, y));
                    y = ld16w(b2 + k); c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(x0, y)); c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(x1, y));
                    y = ld16w(b3 + k); c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(x0, y)); c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(x1, y));
                }
                int32_t s00 = hsum(c00), s01 = hsum(c01), s02 = hsum(c02), s03 = hsum(c03);
                int32_t s10 = hsum(c10), s11 = hsum(c11), s12 = hsum(c12), s13 = hsum(c13);
                for (int k = K16; k < K; k++) {
                    const int w0 = b0[k], w1 = b1[k], w2 = b2[k], w3 = b3[k];
                    s00 += a0[k] * w0; s01 += a0[k] * w1; s02 += a0[k] * w2; s03 += a0[k] * w3;
                    s10 += a1[k] * w0; s11 += a1[k] * w1; s12 += a1[k] * w2; s13 += a1[k] * w3;
                }
                int32_t *c0 = C + (size_t)i * N + jb + j, *c1 = c0 + N;
                c0[0] = s00; c0[1] = s01; c0[2] = s02; c0[3] = s03;
                c1[0] = s10; c1[1] = s11; c1[2] = s12; c1[3] = s13;
            }
            for (; j < nb; j++) {                          /* leftover columns of the last chunk */
                const int16_t *b = W + (size_t)j * Kp;
                C[(size_t)i * N + jb + j] = dot1(a0, b, K);
                C[(size_t)(i + 1) * N + jb + j] = dot1(a1, b, K);
            }
        }
        for (int i = M2; i < M; i++) {                     /* leftover row (M odd) */
            const int8_t *a = A + (size_t)i * K;
            for (int j = 0; j < nb; j++) C[(size_t)i * N + jb + j] = dot1(a, W + (size_t)j * Kp, K);
        }
    }
}
