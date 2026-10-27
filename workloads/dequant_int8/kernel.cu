// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void dequant_int8(const signed char *q, const float *s, float *y, int R, int C)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)R * C)
        return;
    int r = (int)(i / C);
    y[i] = (float)q[i] * s[r];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0] * p[1];
    dequant_int8<<<(unsigned)((n + 255) / 256), 256>>>((const signed char *)buf[0], (const float *)buf[1],
                                                       (float *)buf[2], (int)p[0], (int)p[1]);
}
