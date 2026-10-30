/* Shared timing helpers: min-of-N, batch calibrated to ~20 ms. */
#ifndef TIMING_H
#define TIMING_H

#include <time.h>

static inline double now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

/* Times `stmt` (no top-level commas) and stores the best per-iteration ns. */
#define BENCH(result, stmt)                                              \
    do {                                                                 \
        long it_ = 1;                                                    \
        double t_;                                                       \
        for (;;) {                                                       \
            double t0_ = now_ns();                                       \
            for (long i_ = 0; i_ < it_; i_++) { stmt; }                  \
            t_ = now_ns() - t0_;                                         \
            if (t_ > 2e7) break;                                         \
            it_ *= 2;                                                    \
        }                                                                \
        double best_ = t_ / (double)it_;                                 \
        for (int r_ = 0; r_ < 5; r_++) {                                 \
            double t0_ = now_ns();                                       \
            for (long i_ = 0; i_ < it_; i_++) { stmt; }                  \
            double tt_ = (now_ns() - t0_) / (double)it_;                 \
            if (tt_ < best_) best_ = tt_;                                \
        }                                                                \
        (result) = best_;                                                \
    } while (0)

#endif
