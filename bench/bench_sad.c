/*
 * SAD experiments: is a 16x16 SAD load-bound, cache-bound, or neither?
 *
 *   E1  Load-bound check.  Everything in L1. If sad_x4 (loads the current
 *       block once for four candidates) is faster per SAD than four single
 *       SADs, the single-SAD kernel was limited by loads, not by cache.
 *   E2  Working-set sweep. SADs at random positions in a frame of growing
 *       size. If the kernel becomes cache/memory-bound, ns per SAD should
 *       step up as the frame outgrows L1, L2, L3.
 *   E3  Stride experiment. The SAME block, SAME work, different row strides.
 *       Power-of-two strides map the 16 rows to the same L1 sets, which can
 *       cause conflict misses even though only 16 cache lines are involved.
 *
 * Notes on method:
 *  - Every SAD is a real call into libdsp (as with x264's function-pointer
 *    design). Call overhead is included, and it favours sad_x4 slightly,
 *    since that makes one call per four SADs.
 *  - Min-of-N timing. Pin the process:  taskset -c 2 ./bench_sad
 *  - Results are specific to this machine; they say nothing about others.
 */
#define _POSIX_C_SOURCE 200809L
#include "dsp.h"
#include "timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile uint32_t g_sink;

static uint32_t xs = 2463534242u;
static inline uint32_t xorshift(void)
{
    xs ^= xs << 13; xs ^= xs >> 17; xs ^= xs << 5;
    return xs;
}

static void fill_random(uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)xorshift();
}

/* ---- cache geometry from sysfs (best effort; absent in some sandboxes) --- */
static int read_sys(int idx, const char *name, char *buf, size_t n)
{
    char path[160];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/%s", idx, name);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    if (!fgets(buf, (int)n, f)) { fclose(f); return 0; }
    fclose(f);
    buf[strcspn(buf, "\n")] = 0;
    return 1;
}

static void print_cache_info(void)
{
    printf("Cache geometry (cpu0, from sysfs):\n");
    int any = 0;
    for (int i = 0; i < 8; i++) {
        char lvl[32], type[32], size[32], ways[32];
        if (!read_sys(i, "level", lvl, sizeof lvl)) break;
        read_sys(i, "type", type, sizeof type);
        read_sys(i, "size", size, sizeof size);
        if (!read_sys(i, "ways_of_associativity", ways, sizeof ways)) strcpy(ways, "?");
        printf("  L%s %-12s %8s  %s-way\n", lvl, type, size, ways);
        any = 1;
    }
    if (!any) printf("  (unavailable)\n");
}

typedef uint32_t (*sad_fn)(const uint8_t *, int, const uint8_t *, int);

/* ---- E1 -------------------------------------------------------------------*/
static void e1_load_bound(void)
{
    enum { S = 64 };
    uint8_t *ref = malloc(S * S);
    uint8_t *cur = malloc(16 * 16);
    fill_random(ref, S * S);
    fill_random(cur, 16 * 16);
    const uint8_t *b = ref + 8 * S + 8;   /* 4 candidates one pixel apart */
    uint32_t acc = 0, out[4];

    double t_ref, t_av, t_x2, t_x4;
    BENCH(t_ref, { for (int c = 0; c < 4; c++) acc += sad_16x16_ref(cur, 16, b + c, S); });
    BENCH(t_av,  { for (int c = 0; c < 4; c++) acc += sad_16x16_autovec(cur, 16, b + c, S); });
    BENCH(t_x2,  { for (int c = 0; c < 4; c++) acc += sad_16x16_avx2(cur, 16, b + c, S); });
    BENCH(t_x4,  { sad_x4_16x16_avx2(cur, 16, b, b + 1, b + 2, b + 3, S, out); acc += out[0] + out[3]; });
    g_sink = acc;

    printf("\n## E1: load-bound check (all data in L1: 64x64 frame = 4 KB)\n\n");
    printf("| variant                  | ns per SAD | speedup vs scalar | 16-B loads per SAD |\n");
    printf("|--------------------------|-----------:|------------------:|-------------------:|\n");
    printf("| scalar ref               | %10.2f | %16.2fx | %18s |\n", t_ref / 4, 1.0, "n/a (scalar)");
    printf("| autovec                  | %10.2f | %16.2fx | %18s |\n", t_av / 4, t_ref / t_av, "compiler");
    printf("| avx2 single              | %10.2f | %16.2fx | %18d |\n", t_x2 / 4, t_ref / t_x2, 32);
    printf("| avx2 x4 (cur reused)     | %10.2f | %16.2fx | %18d |\n", t_x4 / 4, t_ref / t_x4, 20);
    /* Control: x4 makes one call per four SADs, so part of its gain could be
     * plain call overhead. Measure an empty library call and subtract it. */
    double t_call;
    BENCH(t_call, fft_plan_destroy(NULL));   /* returns immediately */
    const double k_single = t_x2 / 4 - t_call;
    const double k_x4 = (t_x4 - t_call) / 4;
    printf("\nraw x4 vs single AVX2: %.2fx per SAD.\n", t_x2 / t_x4);
    printf("Empty library call costs %.2f ns (included once per call in every row above).\n", t_call);
    printf("Call-overhead-adjusted kernel time per SAD: single %.2f ns, x4 %.2f ns -> x4 is %.2fx faster.\n",
           k_single, k_x4, k_single / k_x4);
    free(ref); free(cur);
}

/* ---- E2 -------------------------------------------------------------------*/
static uint32_t probe(const uint8_t *cur, const uint8_t *frame, int S, int n)
{
    uint32_t acc = 0;
    const uint32_t span = (uint32_t)(S - 16);
    for (int i = 0; i < n; i++) {
        uint32_t r = xorshift();
        uint32_t x = ((r >> 16) * span) >> 16;
        uint32_t y = ((r & 0xffffu) * span) >> 16;
        acc += sad_16x16_avx2(cur, 16, frame + (size_t)y * S + x, S);
    }
    return acc;
}

static void e2_working_set(void)
{
    printf("\n## E2: working-set sweep (SAD at random positions in an SxS frame)\n\n");
    printf("| frame      | size     | ns per SAD | vs 64x64 |\n");
    printf("|------------|---------:|-----------:|---------:|\n");
    uint8_t cur[16 * 16];
    fill_random(cur, sizeof cur);
    double base = 0;
    for (int S = 64; S <= 8192; S *= 2) {
        uint8_t *frame = malloc((size_t)S * S);
        fill_random(frame, (size_t)S * S);
        const int n = 1 << 14;
        uint32_t acc = 0;
        double t;
        BENCH(t, acc += probe(cur, frame, S, n));
        g_sink = acc;
        const double ns = t / n;
        if (S == 64) base = ns;
        char sz[32];
        double bytes = (double)S * S;
        if (bytes >= 1048576) snprintf(sz, sizeof sz, "%.0f MB", bytes / 1048576);
        else snprintf(sz, sizeof sz, "%.0f KB", bytes / 1024);
        printf("| %5dx%-5d | %8s | %10.2f | %7.2fx |\n", S, S, sz, ns, ns / base);
        free(frame);
    }
}

/* ---- E3 -------------------------------------------------------------------*/
static void e3_stride(void)
{
    printf("\n## E3: stride experiment (one block, repeated; only the row stride changes)\n\n");
    printf("| row stride | ns per SAD | vs stride+64 |\n");
    printf("|-----------:|-----------:|-------------:|\n");
    uint8_t cur[16 * 16];
    fill_random(cur, sizeof cur);
    const int strides[] = {1024, 2048, 4096, 8192};
    for (size_t k = 0; k < sizeof strides / sizeof *strides; k++) {
        double t[2];
        for (int v = 0; v < 2; v++) {
            const int st = strides[k] + (v ? 64 : 0);
            size_t bytes = (size_t)16 * st + 16;
            bytes = (bytes + 4095) & ~(size_t)4095;
            uint8_t *ref = aligned_alloc(4096, bytes);
            fill_random(ref, bytes);
            uint32_t acc = 0;
            BENCH(t[v], acc += sad_16x16_avx2(cur, 16, ref, st));
            g_sink = acc;
            free(ref);
        }
        printf("| %10d | %10.2f | %12s |\n", strides[k], t[0], "");
        printf("| %10d | %10.2f | %11.2fx |\n", strides[k] + 64, t[1], t[0] / t[1]);
    }
    printf("\n'vs stride+64' > 1 means the power-of-two stride is slower than its +64 neighbour.\n");
}

int main(void)
{
    if (!dsp_cpu_has_avx2()) {
        printf("This machine has no AVX2+FMA; bench_sad needs it.\n");
        return 0;
    }
    printf("# SAD experiments (16x16 blocks, 8-bit pixels)\n\n");
    print_cache_info();
    e1_load_bound();
    e2_working_set();
    e3_stride();
    return 0;
}
