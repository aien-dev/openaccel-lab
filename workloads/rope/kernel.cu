// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// one thread per rotated pair; n = T*H*h pairs
__global__ void rope(const float *x, const float *cs, const float *sn, float *y, long long n, long long h,
                     long long H)
{
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) {
        long long i = g % h;
        long long th = g / h;
        long long t = th / H;
        long long o = th * 2 * h + i;
        float c = cs[t * h + i], s = sn[t * h + i];
        float x1 = x[o], x2 = x[o + h];
        y[o] = x1 * c - x2 * s;
        y[o + h] = x1 * s + x2 * c;
    }
}

void oal_launch(const long long *p, void *const *buf)
{
    long long T = p[0], H = p[1], h = p[3];
    long long n = T * H * h;
    rope<<<(unsigned)((n + 255) / 256), 256>>>((const float *)buf[0], (const float *)buf[1],
                                               (const float *)buf[2], (float *)buf[3], n, h, H);
}
