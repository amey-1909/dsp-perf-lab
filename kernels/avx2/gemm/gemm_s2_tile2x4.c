/* Step 2: 2 rows x 4 columns register tile (A widened once per 4 columns, Bt once per 2 rows) */
#include "dsp.h"
#include "gemm_steps.h"
#include <immintrin.h>

static inline __m256i ld16(const int8_t *p)
{
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p));
}

static inline int32_t hsum(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

/* one row . one column; used for every leftover row / column */
static int32_t dot1(const int8_t *a, const int8_t *b, int K)
{
    const int K16 = K & ~15;
    __m256i acc = _mm256_setzero_si256();
    for (int k = 0; k < K16; k += 16) acc = _mm256_add_epi32(acc, _mm256_madd_epi16(ld16(a + k), ld16(b + k)));
    int32_t s = hsum(acc);
    for (int k = K16; k < K; k++) s += (int32_t)a[k] * (int32_t)b[k];
    return s;
}

void gemm_s8_avx2_s2(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    const int M2 = M & ~1, N4 = N & ~3, K16 = K & ~15;

    for (int i = 0; i < M2; i += 2) {
        const int8_t *a0 = A + (size_t)i * K, *a1 = a0 + K;
        int j = 0;
        for (; j < N4; j += 4) {
            const int8_t *b0 = Bt + (size_t)j * K, *b1 = b0 + K, *b2 = b1 + K, *b3 = b2 + K;
            __m256i c00 = _mm256_setzero_si256(), c01 = c00, c02 = c00, c03 = c00;
            __m256i c10 = c00, c11 = c00, c12 = c00, c13 = c00;
            for (int k = 0; k < K16; k += 16) {
                const __m256i x0 = ld16(a0 + k), x1 = ld16(a1 + k);
                __m256i y;
                y = ld16(b0 + k); c00 = _mm256_add_epi32(c00, _mm256_madd_epi16(x0, y)); c10 = _mm256_add_epi32(c10, _mm256_madd_epi16(x1, y));
                y = ld16(b1 + k); c01 = _mm256_add_epi32(c01, _mm256_madd_epi16(x0, y)); c11 = _mm256_add_epi32(c11, _mm256_madd_epi16(x1, y));
                y = ld16(b2 + k); c02 = _mm256_add_epi32(c02, _mm256_madd_epi16(x0, y)); c12 = _mm256_add_epi32(c12, _mm256_madd_epi16(x1, y));
                y = ld16(b3 + k); c03 = _mm256_add_epi32(c03, _mm256_madd_epi16(x0, y)); c13 = _mm256_add_epi32(c13, _mm256_madd_epi16(x1, y));
            }
            int32_t s00 = hsum(c00), s01 = hsum(c01), s02 = hsum(c02), s03 = hsum(c03);
            int32_t s10 = hsum(c10), s11 = hsum(c11), s12 = hsum(c12), s13 = hsum(c13);
            for (int k = K16; k < K; k++) {          /* scalar K tail */
                s00 += a0[k] * b0[k]; s01 += a0[k] * b1[k]; s02 += a0[k] * b2[k]; s03 += a0[k] * b3[k];
                s10 += a1[k] * b0[k]; s11 += a1[k] * b1[k]; s12 += a1[k] * b2[k]; s13 += a1[k] * b3[k];
            }
            int32_t *c0 = C + (size_t)i * N + j, *c1 = c0 + N;
            c0[0] = s00; c0[1] = s01; c0[2] = s02; c0[3] = s03;
            c1[0] = s10; c1[1] = s11; c1[2] = s12; c1[3] = s13;
        }
        for (; j < N; j++) {                         /* leftover columns */
            const int8_t *b = Bt + (size_t)j * K;
            C[(size_t)i * N + j] = dot1(a0, b, K);
            C[(size_t)(i + 1) * N + j] = dot1(a1, b, K);
        }
    }
    for (int i = M2; i < M; i++) {                   /* leftover row (M odd) */
        const int8_t *a = A + (size_t)i * K;
        for (int j = 0; j < N; j++) C[(size_t)i * N + j] = dot1(a, Bt + (size_t)j * K, K);
    }
}