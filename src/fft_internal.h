/* Private FFT plan layout, shared by the scalar reference and the SIMD kernels. */
#ifndef FFT_INTERNAL_H
#define FFT_INTERNAL_H

struct fft_plan {
    int n;
    int *rev;   /* bit-reversal permutation */
    float *tw;  /* n/2 twiddles, interleaved (cos, -sin) of 2*pi*k/n */
};

#endif
