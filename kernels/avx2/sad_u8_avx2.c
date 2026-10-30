/*
 * AVX2 SAD, 16x16 blocks.
 *
 * _mm256_sad_epu8 computes |a-b| for 32 byte pairs and sums each group of 8
 * into a 64-bit lane: no widening step is needed (unlike the int8 GEMM). A
 * 16-byte row is half a vector, so two rows are packed per 256-bit register
 * (2 x 128-bit loads + insert); 8 iterations cover the block.
 *
 * Loads per 16x16 SAD: 4 per iteration x 8 = 32 (16-byte loads).
 *
 * sad_x4 keeps the current block's rows in registers across all four
 * candidates: per iteration 2 (cur) + 4 x 2 (candidates) = 10 loads for four
 * SADs instead of 16, i.e. 20 loads per SAD instead of 32. This is the same
 * "reuse each load" idea as the 2x4 GEMM tile.
 */
#include "dsp.h"
#include <immintrin.h>

/* Two 16-byte rows (p and p+stride) in one 256-bit register. */
static inline __m256i ld2rows(const uint8_t *p, int stride)
{
    __m128i lo = _mm_loadu_si128((const __m128i *)p);
    __m128i hi = _mm_loadu_si128((const __m128i *)(p + stride));
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* Sum the four 64-bit partial sums (each <= 8*255*... fits easily in 32 bits). */
static inline uint32_t hsum_sad(__m256i acc)
{
    __m128i s = _mm_add_epi64(_mm256_castsi256_si128(acc),
                              _mm256_extracti128_si256(acc, 1));
    s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
    return (uint32_t)_mm_cvtsi128_si32(s);
}

uint32_t sad_16x16_avx2(const uint8_t *cur, int cs, const uint8_t *ref, int rs)
{
    __m256i acc = _mm256_setzero_si256();
    for (int y = 0; y < 16; y += 2) {
        __m256i c = ld2rows(cur + (size_t)y * cs, cs);
        __m256i r = ld2rows(ref + (size_t)y * rs, rs);
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(c, r));
    }
    return hsum_sad(acc);
}

void sad_x4_16x16_avx2(const uint8_t *cur, int cs, const uint8_t *r0, const uint8_t *r1,
                       const uint8_t *r2, const uint8_t *r3, int rs, uint32_t out[4])
{
    __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
    for (int y = 0; y < 16; y += 2) {
        __m256i c = ld2rows(cur + (size_t)y * cs, cs);   /* loaded once, used 4x */
        a0 = _mm256_add_epi64(a0, _mm256_sad_epu8(c, ld2rows(r0 + (size_t)y * rs, rs)));
        a1 = _mm256_add_epi64(a1, _mm256_sad_epu8(c, ld2rows(r1 + (size_t)y * rs, rs)));
        a2 = _mm256_add_epi64(a2, _mm256_sad_epu8(c, ld2rows(r2 + (size_t)y * rs, rs)));
        a3 = _mm256_add_epi64(a3, _mm256_sad_epu8(c, ld2rows(r3 + (size_t)y * rs, rs)));
    }
    out[0] = hsum_sad(a0);
    out[1] = hsum_sad(a1);
    out[2] = hsum_sad(a2);
    out[3] = hsum_sad(a3);
}
