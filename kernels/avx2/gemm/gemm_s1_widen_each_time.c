/* Step 1: baseline, 1 row x 4 columns, widen every time (scaffold kernel) */
/*
 * AVX2 int8 GEMM, B pre-transposed.
 *
 * Inner loop: 16 int8 -> sign-extend to 16 x int16 -> _mm256_madd_epi16
 * (multiplies int16 pairs and adds adjacent products into 8 x int32).
 * Products are at most 2 * 128 * 128 = 32768, so madd can never overflow;
 * the int32 accumulators are safe for any practical K.
 *
 * Note: the faster x86 route is VNNI (_mm256_dpbusd_epi32) or the
 * maddubs trick, but those need unsigned x signed operands. madd_epi16 is
 * exact for signed x signed, which keeps this version bit-exact vs the ref.
 *
 * Register blocking: 1 row of A x 4 columns of Bt. The widened A vector is
 * loaded and converted once and reused for 4 dot products.
 *
 * Optional: build with -DMCA_MARKERS and run scripts/mca.sh to feed the
 * inner loop to llvm-mca for a pipeline/port-pressure analysis.
 */
#include "dsp.h"
#include "gemm_steps.h"
#include <immintrin.h>

#ifdef MCA_MARKERS
#define MCA_BEGIN(n) __asm__ volatile("# LLVM-MCA-BEGIN " #n)
#define MCA_END(n)   __asm__ volatile("# LLVM-MCA-END " #n)
#else
#define MCA_BEGIN(n)
#define MCA_END(n)
#endif

static inline int32_t hsum_epi32(__m256i v)
{
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

static inline __m256i ld_s8_to_s16(const int8_t *p)
{
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p));
}

void gemm_s8_avx2_s1(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    const int K16 = K & ~15;

    for (int i = 0; i < M; i++) {
        const int8_t *a = A + (size_t)i * K;
        int32_t *c = C + (size_t)i * N;
        int j = 0;

        for (; j + 4 <= N; j += 4) {
            const int8_t *b0 = Bt + (size_t)j * K;
            const int8_t *b1 = b0 + K;
            const int8_t *b2 = b1 + K;
            const int8_t *b3 = b2 + K;
            __m256i c0 = _mm256_setzero_si256();
            __m256i c1 = _mm256_setzero_si256();
            __m256i c2 = _mm256_setzero_si256();
            __m256i c3 = _mm256_setzero_si256();

            for (int k = 0; k < K16; k += 16) {
                MCA_BEGIN(gemm_inner);
                __m256i av = ld_s8_to_s16(a + k);
                c0 = _mm256_add_epi32(c0, _mm256_madd_epi16(av, ld_s8_to_s16(b0 + k)));
                c1 = _mm256_add_epi32(c1, _mm256_madd_epi16(av, ld_s8_to_s16(b1 + k)));
                c2 = _mm256_add_epi32(c2, _mm256_madd_epi16(av, ld_s8_to_s16(b2 + k)));
                c3 = _mm256_add_epi32(c3, _mm256_madd_epi16(av, ld_s8_to_s16(b3 + k)));
                MCA_END(gemm_inner);
            }

            int32_t s0 = hsum_epi32(c0), s1 = hsum_epi32(c1);
            int32_t s2 = hsum_epi32(c2), s3 = hsum_epi32(c3);
            for (int k = K16; k < K; k++) {
                int32_t av = a[k];
                s0 += av * b0[k];
                s1 += av * b1[k];
                s2 += av * b2[k];
                s3 += av * b3[k];
            }
            c[j] = s0; c[j + 1] = s1; c[j + 2] = s2; c[j + 3] = s3;
        }

        for (; j < N; j++) {
            const int8_t *b = Bt + (size_t)j * K;
            __m256i acc = _mm256_setzero_si256();
            for (int k = 0; k < K16; k += 16)
                acc = _mm256_add_epi32(acc,
                        _mm256_madd_epi16(ld_s8_to_s16(a + k), ld_s8_to_s16(b + k)));
            int32_t s = hsum_epi32(acc);
            for (int k = K16; k < K; k++)
                s += (int32_t)a[k] * b[k];
            c[j] = s;
        }
    }
}
