// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// One block of 32 threads per row; max and sum by a shared-memory tree, every thread reaches every barrier.
__global__ void tsoftmax16(const float *x, float *y)
{
    __shared__ float sh[32];
    long long r = blockIdx.x;
    int j = threadIdx.x;
    float v = x[r * 32 + j];
    sh[j] = v;
    __syncthreads();
    for (int w = 16; w > 0; w /= 2) {
        float o = 0.0f;
        if (j < w)
            o = fmaxf(sh[j], sh[j + w]);
        __syncthreads();
        if (j < w)
            sh[j] = o;
        __syncthreads();
    }
    float m = sh[0];
    __syncthreads();
    float e = expf(v - m);
    sh[j] = e;
    __syncthreads();
    for (int w = 16; w > 0; w /= 2) {
        float o = 0.0f;
        if (j < w)
            o = sh[j] + sh[j + w];
        __syncthreads();
        if (j < w)
            sh[j] = o;
        __syncthreads();
    }
    y[r * 32 + j] = e / sh[0];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long tiles = p[0];
    tsoftmax16<<<(unsigned)(tiles * 16), 32>>>((const float *)buf[0], (float *)buf[1]);
}
