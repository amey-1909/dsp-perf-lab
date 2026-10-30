/* Rung 2 of the ladder: reference source, compiled -O3 -mavx2 -mfma. */
#define SAD_16X16_NAME sad_16x16_autovec
#define SAD_X4_NAME    sad_x4_16x16_autovec
#include "../ref/sad_u8_ref.c"
