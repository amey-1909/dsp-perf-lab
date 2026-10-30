/* Scalar SAD reference. Also #included by kernels/autovec under different
 * names and compiler flags, so both stages share one source. */
#include "dsp.h"

#ifndef SAD_16X16_NAME
#define SAD_16X16_NAME sad_16x16_ref
#define SAD_X4_NAME    sad_x4_16x16_ref
#endif

static inline uint32_t sad16_scalar(const uint8_t *cur, int cs, const uint8_t *ref, int rs)
{
    uint32_t s = 0;
    for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
            int d = (int)cur[y * cs + x] - (int)ref[y * rs + x];
            s += (uint32_t)(d < 0 ? -d : d);
        }
    }
    return s;
}

uint32_t SAD_16X16_NAME(const uint8_t *cur, int cs, const uint8_t *ref, int rs)
{
    return sad16_scalar(cur, cs, ref, rs);
}

void SAD_X4_NAME(const uint8_t *cur, int cs, const uint8_t *r0, const uint8_t *r1,
                 const uint8_t *r2, const uint8_t *r3, int rs, uint32_t out[4])
{
    out[0] = sad16_scalar(cur, cs, r0, rs);
    out[1] = sad16_scalar(cur, cs, r1, rs);
    out[2] = sad16_scalar(cur, cs, r2, rs);
    out[3] = sad16_scalar(cur, cs, r3, rs);
}
