/* Rung 2 of the ladder: reference source, compiled -O3 -mavx2 -mfma. */
#define GEMM_S8_NAME gemm_s8_autovec
#include "../ref/gemm_s8_ref.c"
