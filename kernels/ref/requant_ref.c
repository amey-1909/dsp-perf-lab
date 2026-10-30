/* Fixed-point requantization, bit-exact with the gemmlowp / TFLite reference.
 * This is the step NPUs and Hexagon HVX do after an int8 MAC array. */
#include "dsp.h"

/* SaturatingRoundingDoublingHighMul: round(a * b / 2^31) */
static inline int32_t srdhm(int32_t a, int32_t b)
{
    if (a == INT32_MIN && b == INT32_MIN)
        return INT32_MAX;
    int64_t ab = (int64_t)a * (int64_t)b;
    int32_t nudge = ab >= 0 ? (1 << 30) : (1 - (1 << 30));
    return (int32_t)((ab + nudge) / (1LL << 31));
}

/* RoundingDivideByPowerOfTwo: round-half-away-from-zero x / 2^exp */
static inline int32_t rdivpot(int32_t x, int exp)
{
    int32_t mask = (int32_t)((1u << exp) - 1u);
    int32_t rem = x & mask;
    int32_t thr = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> exp) + (rem > thr ? 1 : 0);
}

void requant_s32_to_s8_ref(const int32_t *acc, int8_t *out, size_t n,
                           int32_t mult, int shift, int32_t zero_point)
{
    for (size_t i = 0; i < n; i++) {
        int32_t v = rdivpot(srdhm(acc[i], mult), shift) + zero_point;
        if (v > 127) v = 127;
        if (v < -128) v = -128;
        out[i] = (int8_t)v;
    }
}
