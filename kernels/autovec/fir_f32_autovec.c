/* Rung 2 of the ladder: identical source to the scalar reference, compiled
 * with -O3 -mavx2 -mfma. Whether GCC vectorizes it is the point of study:
 * run scripts/vec_report.sh to see the compiler's own explanation. */
#define FIR_F32_NAME fir_f32_autovec
#include "../ref/fir_f32_ref.c"
