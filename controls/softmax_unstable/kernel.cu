// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>
#include <math.h>

#define NT 128

// one block of NT threads per row
// CONTROL: the max subtraction is deliberately omitted
__global__ void softmax(const float *x, float *y, long long D)
{
    __shared__ float red[NT];
    int t = threadIdx.x;
    const float *xr = x + (long long)blockIdx.x * D;
    float *yr = y + (long long)blockIdx.x * D;
    float acc = 0.0f;
    for (long long k = t; k < D; k += NT) {
        float e = expf(xr[k]);
        yr[k] = e;
        acc += e;
    }
    red[t] = acc;
    __syncthreads();
    for (int h = NT / 2; h > 0; h >>= 1) {
        if (t < h)
            red[t] += red[t + h];
        __syncthreads();
    }
    float s = red[0];
    for (long long k = t; k < D; k += NT)
        yr[k] = yr[k] / s;
}

void oal_launch(const long long *p, void *const *buf)
{
    softmax<<<(unsigned)p[0], NT>>>((const float *)buf[0], (float *)buf[1], p[1]);
}
