// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void vecadd(const float *a, const float *b, float *y, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a[i] + b[i];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0];
    vecadd<<<(unsigned)((n + 255) / 256), 256>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], n);
}
