// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void silu(const float *x, float *y, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0];
    silu<<<(unsigned)((n + 255) / 256), 256>>>((const float *)buf[0], (float *)buf[1], n);
}
