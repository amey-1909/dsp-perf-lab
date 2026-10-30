cmake_minimum_required(VERSION 3.16)
project(dsp_perf_lab C)

set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3 -g")   # -g so perf annotate / llvm-mca have symbols

option(DSP_SANITIZE "Build with ASan + UBSan" OFF)
option(DSP_COVERAGE "Build with gcov instrumentation" OFF)

add_compile_options(-Wall -Wextra)
if(DSP_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
if(DSP_COVERAGE)
  add_compile_options(--coverage)
  add_link_options(--coverage)
endif()

set(REF_SRCS
  kernels/ref/fir_f32_ref.c
  kernels/ref/gemm_s8_ref.c
  kernels/ref/requant_ref.c
  kernels/ref/fft_f32_ref.c
  kernels/ref/sad_u8_ref.c)
set(AUTOVEC_SRCS
  kernels/autovec/fir_f32_autovec.c
  kernels/autovec/gemm_s8_autovec.c
  kernels/autovec/sad_u8_autovec.c)
set(AVX2_SRCS
  kernels/avx2/fir_f32_avx2.c
  kernels/avx2/gemm_s8_avx2.c
  kernels/avx2/sad_u8_avx2.c)

# FFT step ladder: every kernels/avx2/fft/*.c is picked up automatically, so a
# new step needs no CMake edit.
file(GLOB FFT_STEP_SRCS CONFIGURE_DEPENDS kernels/avx2/fft/*.c)
list(APPEND AVX2_SRCS ${FFT_STEP_SRCS})

add_library(dsp SHARED ${REF_SRCS} ${AUTOVEC_SRCS} ${AVX2_SRCS} src/cpu.c)
target_include_directories(dsp PUBLIC include PRIVATE src)
target_link_libraries(dsp PUBLIC m)

# Rung 1: true scalar baseline. Auto-vectorization off so "ref" means scalar.
set_source_files_properties(${REF_SRCS} PROPERTIES
  COMPILE_OPTIONS "-fno-tree-vectorize;-fno-tree-slp-vectorize")
# Rung 2: same source, compiler is free to vectorize for AVX2.
set_source_files_properties(${AUTOVEC_SRCS} ${AVX2_SRCS} PROPERTIES
  COMPILE_OPTIONS "-mavx2;-mfma")

add_executable(test_kernels tests/test_kernels.c)
target_link_libraries(test_kernels PRIVATE dsp)

add_executable(bench bench/bench.c)
target_link_libraries(bench PRIVATE dsp)

add_executable(bench_sad bench/bench_sad.c)
target_link_libraries(bench_sad PRIVATE dsp)

add_executable(bench_fft bench/bench_fft.c)
target_link_libraries(bench_fft PRIVATE dsp)

enable_testing()
add_test(NAME kernels COMMAND test_kernels)
