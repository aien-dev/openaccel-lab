// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// One thread per row: absmax, scale = absmax / 127, q = clamp(rint(x / scale), -127, 127).
__global__ void quant_int8(const float *x, signed char *q, float *s, int R, int C)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R)
        return;
    float amax = 0.0f;
    for (int c = 0; c < C; c++)
        amax = fmaxf(amax, fabsf(x[r * C + c]));
    float sc = amax / 127.0f;
    s[r] = sc;
    for (int c = 0; c < C; c++) {
        float y = sc > 0.0f ? x[r * C + c] / sc : 0.0f;
        float v = fminf(fmaxf(rintf(y), -127.0f), 127.0f);
        q[r * C + c] = (signed char)(int)v;
    }
}

void oal_launch(const long long *p, void *const *buf)
{
    int R = (int)p[0];
    int C = (int)p[1];
    quant_int8<<<(R + 63) / 64, 64>>>((const float *)buf[0], (signed char *)buf[1], (float *)buf[2], R, C);
}
