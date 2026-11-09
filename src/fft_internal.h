/* Private FFT plan layout, shared by the scalar reference and the SIMD kernels. */
#ifndef FFT_INTERNAL_H
#define FFT_INTERNAL_H

struct fft_plan {
    int n;
    int *rev;   /* bit-reversal permutation */
    float *tw;  /* n/2 twiddles, interleaved (cos, -sin) of 2*pi*k/n */
    float *tw_re;   /* per-stage table, real part: index half + j */
    float *tw_im;   /* per-stage table, imaginary part */
    
};

#endif
