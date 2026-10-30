/* Scalar int8 GEMM reference (B pre-transposed). Shared with kernels/autovec. */
#include "dsp.h"

#ifndef GEMM_S8_NAME
#define GEMM_S8_NAME gemm_s8_ref
#endif

void GEMM_S8_NAME(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    for (int i = 0; i < M; i++) {
        const int8_t *a = A + (size_t)i * K;
        for (int j = 0; j < N; j++) {
            const int8_t *b = Bt + (size_t)j * K;
            int32_t acc = 0;
            for (int k = 0; k < K; k++)
                acc += (int32_t)a[k] * (int32_t)b[k];
            C[(size_t)i * N + j] = acc;
        }
    }
}
