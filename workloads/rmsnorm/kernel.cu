// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define NT 128

// one block of NT threads per row
__global__ void rmsnorm(const float *x, const float *w, float *y, long long D)
{
    __shared__ float red[NT];
    int t = threadIdx.x;
    const float *xr = x + (long long)blockIdx.x * D;
    float *yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long k = t; k < D; k += NT)
        acc += xr[k] * xr[k];
    red[t] = acc;
    __syncthreads();
    for (int h = NT / 2; h > 0; h >>= 1) {
        if (t < h)
            red[t] += red[t + h];
        __syncthreads();
    }
    float rs = 1.0f / sqrtf(red[0] / (float)D + 1e-6f);
    for (long long k = t; k < D; k += NT)
        yr[k] = xr[k] * rs * w[k];
}

void oal_launch(const long long *p, void *const *buf)
{
    rmsnorm<<<(unsigned)p[0], NT>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], p[1]);
}
