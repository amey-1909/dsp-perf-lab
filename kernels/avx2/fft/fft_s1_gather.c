/* Step 1: baseline AVX2 FFT (gather loads, gather twiddles, scalar stores) */
#include "dsp.h"
#include "fft_internal.h"
#include "fft_steps.h"
#include <immintrin.h>

void fft_f32_avx2_s1(const fft_plan *p, float *d)
{
    const int n = p->n;

    for (int i = 0; i < n; i++) {
        int j = p->rev[i];
        if (i < j) {
            float tr = d[2 * i], ti = d[2 * i + 1];
            d[2 * i] = d[2 * j];     d[2 * i + 1] = d[2 * j + 1];
            d[2 * j] = tr;           d[2 * j + 1] = ti;
        }
    }

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const int step = n / len;

        int i = 0;
        int j;
        for (; i < n; i += len) {
            j = 0;

            __m256i even = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
            __m256i even_tf = _mm256_setr_epi32(0*step, 2*step, 4*step, 6*step, 8*step, 10*step, 12*step, 14*step);
            for (; j <= half-8; j=j+8) {

               __m256 wr = _mm256_i32gather_ps( p->tw+2 * j * step,even_tf, 4);
               __m256 wi = _mm256_i32gather_ps( p->tw+2 * j * step+1,even_tf, 4);

               __m256 b_0 = _mm256_i32gather_ps( d + 2 * (i + j + half),even, 4);
               __m256 b_1 = _mm256_i32gather_ps( d + 2 * (i + j + half)+1,even, 4);

               __m256 a_0 = _mm256_i32gather_ps( d + 2 * (i + j),even, 4);
               __m256 a_1 = _mm256_i32gather_ps( d + 2 * (i + j)+1,even, 4);

               __m256 vr=_mm256_mul_ps(b_0, wr);
               vr=_mm256_fnmadd_ps(b_1, wi,vr);

               __m256 vi=_mm256_mul_ps(b_0, wi);
               vi=_mm256_fmadd_ps(b_1, wr,vi);

               __m256 ur = a_0;
               __m256 ui = a_1;

               __m256 out_a0 = _mm256_add_ps(ur, vr);
               __m256 out_a1 = _mm256_add_ps(ui, vi);

               __m256 out_b0 = _mm256_sub_ps(ur, vr);
               __m256 out_b1 = _mm256_sub_ps(ui, vi);

               float ar[8], ai[8], br[8], bi[8];

               _mm256_storeu_ps(ar, out_a0);
               _mm256_storeu_ps(ai, out_a1);
               _mm256_storeu_ps(br, out_b0);
               _mm256_storeu_ps(bi, out_b1);

               for (int k = 0; k < 8; k++) {
                  d[2 * (i + j + k)] =
                     ar[k];

                  d[2 * (i + j + k) + 1] =
                     ai[k];

                  d[2 * (i + j + k + half)] =
                     br[k];

                  d[2 * (i + j + k + half) + 1] =
                     bi[k];
               }
            }
            for (; j < half; j++) {
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
}
