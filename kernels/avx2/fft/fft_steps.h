/* Private: prototypes of the AVX2 FFT steps. scripts/new_fft_step.sh adds a
 * line above the marker; do not remove the marker. */
#ifndef FFT_STEPS_H
#define FFT_STEPS_H

#include "dsp.h"

void fft_f32_avx2_s1(const fft_plan *p, float *d);
/* NEW PROTOTYPES GO ABOVE THIS LINE */

#endif
