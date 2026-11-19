#!/usr/bin/env python3
"""Cross-check the C kernels against NumPy through ctypes.

Independent of the C test-suite: here the golden model is NumPy/SciPy-class
code, which is the "checked against NumPy for bit-exactness or SNR" rung.

    python3 python/validate.py build/libdsp.so
"""
import ctypes
import sys

import numpy as np

lib_path = sys.argv[1] if len(sys.argv) > 1 else "build/libdsp.so"
lib = ctypes.CDLL(lib_path)

f32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
c64 = np.ctypeslib.ndpointer(np.complex64, flags="C_CONTIGUOUS")
s8 = np.ctypeslib.ndpointer(np.int8, flags="C_CONTIGUOUS")
s32 = np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS")
i = ctypes.c_int

lib.dsp_cpu_has_avx2.restype = i
lib.fft_plan_create.restype = ctypes.c_void_p
lib.fft_plan_create.argtypes = [i]
lib.fft_plan_destroy.argtypes = [ctypes.c_void_p]
lib.fft_f32_ref.argtypes = [ctypes.c_void_p, c64]
lib.fft_avx2_num_steps.restype = i
lib.fft_avx2_step_name.restype = ctypes.c_char_p
lib.fft_avx2_step_name.argtypes = [i]
lib.fft_avx2_step_run.argtypes = [i, ctypes.c_void_p, c64]
u32 = np.ctypeslib.ndpointer(np.uint32, flags="C_CONTIGUOUS")
vp = ctypes.c_void_p
for name in ("ref", "autovec", "avx2"):
    f = getattr(lib, f"sad_16x16_{name}")
    f.argtypes, f.restype = [vp, i, vp, i], ctypes.c_uint32
    getattr(lib, f"sad_x4_16x16_{name}").argtypes = [vp, i, vp, vp, vp, vp, i, u32]
for name in ("ref", "autovec", "avx2"):
    getattr(lib, f"fir_f32_{name}").argtypes = [f32, f32, f32, i, i]
    getattr(lib, f"gemm_s8_{name}").argtypes = [s8, s8, s32, i, i, i]
lib.gemm_avx2_num_steps.restype = i
lib.gemm_avx2_step_name.restype = ctypes.c_char_p
lib.gemm_avx2_step_name.argtypes = [i]
lib.gemm_avx2_step_run.argtypes = [i, s8, s8, s32, i, i, i]

has_avx2 = bool(lib.dsp_cpu_has_avx2())
variants = ["ref"] + (["autovec", "avx2"] if has_avx2 else [])
rng = np.random.default_rng(0)
failures = 0


def report(ok, msg):
    global failures
    failures += not ok
    print(("PASS  " if ok else "FAIL  ") + msg)


# ---- FIR vs np.correlate (float64 golden) ---------------------------------
for n, taps in [(1000, 33), (4096, 64), (8191, 7)]:
    x = rng.uniform(-1, 1, n + taps - 1).astype(np.float32)
    h = rng.uniform(-1, 1, taps).astype(np.float32)
    gold = np.correlate(x.astype(np.float64), h.astype(np.float64), "valid")
    for v in variants:
        y = np.empty(n, np.float32)
        getattr(lib, f"fir_f32_{v}")(x, h, y, n, taps)
        err = np.max(np.abs(y - gold))
        snr = 10 * np.log10(np.sum(gold**2) / np.sum((y - gold) ** 2))
        report(err < 1e-4, f"fir  {v:8s} n={n:5d} taps={taps:2d} max_err={err:.2e} SNR={snr:.0f} dB")

# ---- int8 GEMM vs NumPy int32 matmul (must be bit-exact) ------------------
for M, N, K in [(17, 13, 33), (64, 64, 64), (128, 96, 257)]:
    A = rng.integers(-128, 128, (M, K), dtype=np.int8)
    B = rng.integers(-128, 128, (K, N), dtype=np.int8)
    Bt = np.ascontiguousarray(B.T)
    gold = A.astype(np.int32) @ B.astype(np.int32)
    for v in variants:
        C = np.empty((M, N), np.int32)
        getattr(lib, f"gemm_s8_{v}")(A, Bt, C, M, N, K)
        report(np.array_equal(C, gold), f"gemm {v:8s} {M}x{N}x{K} bit-exact")
    for st in range(lib.gemm_avx2_num_steps() if has_avx2 else 0):
        C = np.empty((M, N), np.int32)
        lib.gemm_avx2_step_run(st, A, Bt, C, M, N, K)
        nm = lib.gemm_avx2_step_name(st).decode()
        report(np.array_equal(C, gold), f"gemm {nm:24s} {M}x{N}x{K} bit-exact")

# ---- SAD vs NumPy abs-diff (must be exact) ------------------------------------
S = 96
bad = {v: 0 for v in variants}
for trial in range(60):
    ref = rng.integers(0, 256, (S, S), dtype=np.uint8)
    cur = rng.integers(0, 256, (S, S), dtype=np.uint8)
    if trial == 0:
        ref[:] = 0
        cur[:] = 255                      # maximum possible SAD = 65280
    pos = rng.integers(0, S - 16 + 1, (5, 2))
    (cy, cx), cands = pos[0], pos[1:]
    cblk = cur[cy:cy + 16, cx:cx + 16].astype(np.int32)
    cur_ptr = cur.ctypes.data + int(cy) * S + int(cx)
    ptrs = [ref.ctypes.data + int(y) * S + int(x) for y, x in cands]
    gold = [int(np.abs(cblk - ref[y:y + 16, x:x + 16].astype(np.int32)).sum()) for y, x in cands]
    for v in variants:
        singles = [getattr(lib, f"sad_16x16_{v}")(cur_ptr, S, p, S) for p in ptrs]
        out = np.empty(4, np.uint32)
        getattr(lib, f"sad_x4_16x16_{v}")(cur_ptr, S, *ptrs, S, out)
        bad[v] += (singles != gold) + (list(out) != gold)
for v in variants:
    report(bad[v] == 0, f"sad  {v:8s} single + x4 exact vs NumPy over 60 trials (incl. max-SAD case)")

# ---- FFT vs np.fft.fft ------------------------------------------------------
for n in (64, 1024, 4096):
    x = (rng.uniform(-1, 1, n) + 1j * rng.uniform(-1, 1, n)).astype(np.complex64)
    gold = np.fft.fft(x.astype(np.complex128))
    plan = lib.fft_plan_create(n)
    runs = [("ref", lambda y: lib.fft_f32_ref(plan, y))]
    if has_avx2:
        for st in range(lib.fft_avx2_num_steps()):
            runs.append((lib.fft_avx2_step_name(st).decode(),
                         lambda y, st=st: lib.fft_avx2_step_run(st, plan, y)))
    for name, run in runs:
        y = x.copy()
        run(y)
        rel = np.linalg.norm(y - gold) / np.linalg.norm(gold)
        report(rel < 1e-5, f"fft  {name:12s} n={n:5d} rel_L2_err={rel:.2e}")
    lib.fft_plan_destroy(plan)

print(f"\n{'ALL PASSED' if not failures else str(failures) + ' FAILED'} (avx2 {'tested' if has_avx2 else 'skipped'})")
sys.exit(1 if failures else 0)
