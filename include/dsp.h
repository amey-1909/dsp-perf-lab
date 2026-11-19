/*
 * dsp-perf-lab: public API.
 *
 * Every kernel exists in several variants so each rung of the optimization
 * ladder can be tested and benchmarked against the one below it:
 *
 *   *_ref      scalar C, auto-vectorization disabled (the correctness oracle)
 *   *_autovec  same source, compiled -O3 -mavx2 -mfma (what the compiler gives you)
 *   *_avx2     hand-written AVX2 intrinsics
 */
#ifndef DSP_H
#define DSP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Runtime check used by tests/bench before calling any *_avx2 kernel. */
int dsp_cpu_has_avx2(void);

/* ---------------------------------------------------------------- FIR ----
 * "Valid" correlation:  y[i] = sum_{k=0..taps-1} h[k] * x[i+k],  i in [0,n)
 * x must hold n + taps - 1 samples. Reverse h for true convolution.
 */
void fir_f32_ref(const float *x, const float *h, float *y, int n, int taps);
void fir_f32_autovec(const float *x, const float *h, float *y, int n, int taps);
void fir_f32_avx2(const float *x, const float *h, float *y, int n, int taps);

/* ------------------------------------------------------------ int8 GEMM --
 * C[M][N] (int32) = A[M][K] (int8) * B[K][N] (int8)
 * B is passed PRE-TRANSPOSED as Bt[N][K] so both operands stream along K
 * (the layout NPUs / DSPs expect for their dot-product instructions).
 * All matrices are row-major and densely packed.
 */
void gemm_s8_ref(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);
void gemm_s8_autovec(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);
/* gemm_s8_avx2 runs the LAST step of the AVX2 ladder (kernels/avx2/gemm/). */
void gemm_s8_avx2(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);

/* AVX2 GEMM step ladder: one optimization per step, each in its own file. Step i is
 * 0-based here; gemm_avx2_step_name(0) is "s1_...". Every step must be bit-exact. */
typedef void (*gemm_fn)(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);
int         gemm_avx2_num_steps(void);
const char *gemm_avx2_step_name(int i);
const char *gemm_avx2_step_desc(int i);
gemm_fn     gemm_avx2_step_fn(int i);
void        gemm_avx2_step_run(int i, const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);

/* ------------------------------------------------------- requantization --
 * gemmlowp / TFLite-style: out = clamp_s8( RDIVPOT( SRDHM(acc, mult), shift ) + zp )
 * mult is a Q31 multiplier, shift is a right shift in [0, 30].
 * Real scale represented = (mult / 2^31) / 2^shift.
 */
void requant_s32_to_s8_ref(const int32_t *acc, int8_t *out, size_t n,
                           int32_t mult, int shift, int32_t zero_point);

/* ----------------------------------------------------------------- FFT ----
 * In-place radix-2 DIT complex FFT, n must be a power of two.
 * data is interleaved: re0, im0, re1, im1, ...
 */
typedef struct fft_plan fft_plan;
fft_plan *fft_plan_create(int n);
void fft_plan_destroy(fft_plan *p);
void fft_f32_ref(const fft_plan *p, float *data);

/* AVX2 FFT step ladder (kernels/avx2/fft/): one optimization per step, each in
 * its own file. Step i is 0-based here; fft_avx2_step_name(0) is "s1_...". */
typedef void (*fft_fn)(const fft_plan *p, float *data);
int         fft_avx2_num_steps(void);
const char *fft_avx2_step_name(int i);
const char *fft_avx2_step_desc(int i);
fft_fn      fft_avx2_step_fn(int i);
void        fft_avx2_step_run(int i, const fft_plan *p, float *data);

/* ----------------------------------------------------------------- SAD ----
 * Sum of absolute differences of a 16x16 block of 8-bit pixels (the core of
 * video motion estimation). cs / rs are row strides in bytes of the current
 * and reference blocks. The kernels read exactly 16 bytes per row.
 *
 * sad_x4: ONE current block against FOUR candidate reference blocks (shared
 * stride rs). The current block is loaded once and reused, as in x264's
 * sad_x4. out[i] is the SAD against r_i.
 */
uint32_t sad_16x16_ref(const uint8_t *cur, int cs, const uint8_t *ref, int rs);
uint32_t sad_16x16_autovec(const uint8_t *cur, int cs, const uint8_t *ref, int rs);
uint32_t sad_16x16_avx2(const uint8_t *cur, int cs, const uint8_t *ref, int rs);

void sad_x4_16x16_ref(const uint8_t *cur, int cs, const uint8_t *r0, const uint8_t *r1,
                      const uint8_t *r2, const uint8_t *r3, int rs, uint32_t out[4]);
void sad_x4_16x16_autovec(const uint8_t *cur, int cs, const uint8_t *r0, const uint8_t *r1,
                          const uint8_t *r2, const uint8_t *r3, int rs, uint32_t out[4]);
void sad_x4_16x16_avx2(const uint8_t *cur, int cs, const uint8_t *r0, const uint8_t *r1,
                       const uint8_t *r2, const uint8_t *r3, int rs, uint32_t out[4]);

#ifdef __cplusplus
}
#endif

#endif /* DSP_H */
