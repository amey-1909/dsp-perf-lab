/* Private: prototypes of the AVX2 int8 GEMM steps. scripts/new_gemm_step.sh adds
 * a line above the marker; do not remove the marker. */
#ifndef GEMM_STEPS_H
#define GEMM_STEPS_H

#include "dsp.h"

void gemm_s8_avx2_s1(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);
void gemm_s8_avx2_s2(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K);
/* NEW PROTOTYPES GO ABOVE THIS LINE */

#endif
