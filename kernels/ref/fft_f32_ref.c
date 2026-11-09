/* Iterative radix-2 DIT FFT, float32, interleaved complex, precomputed plan. */
#include "dsp.h"
#include <math.h>
#include <stdlib.h>

#define DSP_PI 3.14159265358979323846

#include "fft_internal.h"

fft_plan *fft_plan_create(int n)
{
    if (n < 2 || (n & (n - 1)) != 0)
        return NULL;
    fft_plan *p = (fft_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->n = n;
    p->rev = (int *)malloc((size_t)n * sizeof(int));
    p->tw = (float *)malloc((size_t)n * sizeof(float)); /* n/2 complex */    
    p->tw_re = (float *)malloc((size_t)n * sizeof(float));
    p->tw_im = (float *)malloc((size_t)n * sizeof(float));
    if (!p->rev || !p->tw || !p->tw_re || !p->tw_im) { fft_plan_destroy(p); return NULL; }
    int bits = 0;
    while ((1 << bits) < n) bits++;
    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int b = 0; b < bits; b++)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        p->rev[i] = r;
    }
    for (int k = 0; k < n / 2; k++) {
        double a = 2.0 * DSP_PI * (double)k / (double)n;
        p->tw[2 * k]     = (float)cos(a);
        p->tw[2 * k + 1] = (float)-sin(a);
    }
        /* per-stage tables: stage with this `half` needs twiddle j at tw[2*j*step],
     * step = n / (2*half). Same formula as above, so the values are identical. */
    for (int half = 1; half < n; half <<= 1) {
        const int step = n / (2 * half);
        for (int j = 0; j < half; j++) {
            double a = 2.0 * DSP_PI * (double)(j * step) / (double)n;
            p->tw_re[half + j] = (float)cos(a);
            p->tw_im[half + j] = (float)-sin(a);
        }
    }
    return p;
}

void fft_plan_destroy(fft_plan *p)
{   
    if (!p) return;
    free(p->rev);
    free(p->tw);
    free(p->tw_re);
    free(p->tw_im);
    free(p);
}

void fft_f32_ref(const fft_plan *p, float *d)
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
}
