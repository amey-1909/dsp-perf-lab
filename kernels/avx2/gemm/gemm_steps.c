/* Registry of the AVX2 int8 GEMM optimization steps, in order.
 * Tests, the NumPy cross-check and bench_gemm iterate over this table, so a new
 * step is checked (bit-exact against the scalar reference) and timed against
 * the previous step automatically. scripts/new_gemm_step.sh adds a line above
 * the marker. The public function gemm_s8_avx2() runs the LAST step. */
#include "dsp.h"
#include "gemm_steps.h"

typedef struct {
    const char *name;
    const char *desc;
    gemm_fn fn;
} gemm_step;

static const gemm_step steps[] = {
    {"s1_widen_each_time", "1 row x 4 columns, sign-extend operands in the inner loop (baseline)", gemm_s8_avx2_s1},
    {"s2_tile2x4", "2 rows x 4 columns register tile", gemm_s8_avx2_s2},
    {"s3_prepack16", "Bt widened to int16 once up front; inner loop loads B as int16", gemm_s8_avx2_s3},
    {"s4_block_l2", "column blocking: widen a chunk of Bt into a small L2-resident int16 buffer, reuse it for all rows", gemm_s8_avx2_s4},
    /* NEW STEPS GO ABOVE THIS LINE */
};

#define NSTEPS ((int)(sizeof steps / sizeof steps[0]))

int gemm_avx2_num_steps(void) { return NSTEPS; }

const char *gemm_avx2_step_name(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].name : 0; }

const char *gemm_avx2_step_desc(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].desc : 0; }

gemm_fn gemm_avx2_step_fn(int i) { return (i >= 0 && i < NSTEPS) ? steps[i].fn : 0; }

void gemm_avx2_step_run(int i, const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    if (i >= 0 && i < NSTEPS) steps[i].fn(A, Bt, C, M, N, K);
}

/* The public AVX2 GEMM is the newest step. */
void gemm_s8_avx2(const int8_t *A, const int8_t *Bt, int32_t *C, int M, int N, int K)
{
    steps[NSTEPS - 1].fn(A, Bt, C, M, N, K);
}
