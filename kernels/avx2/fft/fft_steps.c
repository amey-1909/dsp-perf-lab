/* Registry of the AVX2 FFT optimization steps, in order.
 * Tests, the NumPy cross-check and bench_fft iterate over this table, so a
 * new step is checked for correctness and timed against the previous one
 * automatically. scripts/new_fft_step.sh adds a line above the marker. */
#include "dsp.h"
#include "fft_steps.h"

typedef struct {
    const char *name;
    const char *desc;
    fft_fn fn;
} fft_step;

static const fft_step steps[] = {
    {"s1_gather", "gather loads and twiddles, scalar stores (baseline)", fft_f32_avx2_s1},
    {"s2_vector_store", "replace the scalar store loop with vector stores", fft_f32_avx2_s2},
    {"s3_twiddle_table", "contiguous per-stage twiddle table, no gathers", fft_f32_avx2_s3},
    /* NEW STEPS GO ABOVE THIS LINE */
};

#define NSTEPS ((int)(sizeof steps / sizeof steps[0]))

int fft_avx2_num_steps(void) { return NSTEPS; }

const char *fft_avx2_step_name(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].name : 0; }

const char *fft_avx2_step_desc(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].desc : 0; }

fft_fn fft_avx2_step_fn(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].fn : 0; }

void fft_avx2_step_run(int i, const fft_plan *p, float *data)
{
    if (i >= 0 && i < NSTEPS) steps[i].fn(p, data);
}
