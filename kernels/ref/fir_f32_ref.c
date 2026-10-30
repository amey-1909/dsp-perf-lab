/* Scalar FIR reference. Also #included by kernels/autovec with a different
 * function name and compiler flags, so both stages share one source. */
#include "dsp.h"

#ifndef FIR_F32_NAME
#define FIR_F32_NAME fir_f32_ref
#endif

void FIR_F32_NAME(const float *x, const float *h, float *y, int n, int taps)
{
    for (int i = 0; i < n; i++) {
        float acc = 0.0f;
        for (int k = 0; k < taps; k++)
            acc += h[k] * x[i + k];
        y[i] = acc;
    }
}
