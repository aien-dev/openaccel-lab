// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void saxpy(const float *alpha, const float *x, const float *y_in, float *y, long long n)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = alpha[0] * x[i] + y_in[i];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0];
    saxpy<<<(unsigned)((n + 255) / 256), 256>>>((const float *)buf[0], (const float *)buf[1],
                                                (const float *)buf[2], (float *)buf[3], n);
}
