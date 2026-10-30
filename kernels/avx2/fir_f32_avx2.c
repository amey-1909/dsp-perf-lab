/*
 * AVX2/FMA FIR.
 *
 * Why this beats the compiler: the scalar form is a float reduction over k.
 * Without -ffast-math GCC may not reassociate it, so (see
 * scripts/vec_report.sh) it vectorizes the multiplies but must still add the
 * lanes one at a time to preserve rounding order, and the gain is small.
 * Here we vectorize ACROSS OUTPUT SAMPLES instead. Each lane owns one output
 * and still accumulates its taps in order k = 0..taps-1, so no
 * reassociation is needed and the whole loop stays in SIMD registers.
 *
 * Register blocking: 4 independent accumulators (32 outputs) hide the 4-5
 * cycle FMA latency. Haswell+ can issue 2 FMAs/cycle, so we need ~8 FMAs in
 * flight; 4 accumulators x 2 is a good start, 8 is a worthwhile experiment.
 * Each broadcast of h[k] is reused 4x, and the x loads are unaligned
 * (x+i+k moves by one float per tap) which modern cores handle at full rate
 * unless the load splits a cache line.
 */
#include "dsp.h"
#include <immintrin.h>

void fir_f32_avx2(const float *x, const float *h, float *y, int n, int taps)
{
    int i = 0;

    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps();
        const float *xp = x + i;
        for (int k = 0; k < taps; k++) {
            __m256 hv = _mm256_broadcast_ss(h + k);
            a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(xp + k),      a0);
            a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(xp + k + 8),  a1);
            a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(xp + k + 16), a2);
            a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(xp + k + 24), a3);
        }
        _mm256_storeu_ps(y + i,      a0);
        _mm256_storeu_ps(y + i + 8,  a1);
        _mm256_storeu_ps(y + i + 16, a2);
        _mm256_storeu_ps(y + i + 24, a3);
    }

    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_setzero_ps();
        for (int k = 0; k < taps; k++)
            a = _mm256_fmadd_ps(_mm256_broadcast_ss(h + k),
                                _mm256_loadu_ps(x + i + k), a);
        _mm256_storeu_ps(y + i, a);
    }

    for (; i < n; i++) {
        float acc = 0.0f;
        for (int k = 0; k < taps; k++)
            acc += h[k] * x[i + k];
        y[i] = acc;
    }
}
