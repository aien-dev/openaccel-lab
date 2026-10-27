// SPDX-License-Identifier: Apache-2.0
// Pass 1: each block of 256 takes the max of its slice in a shared-memory tree.
// Pass 2: one block takes the max of the nb partials into y[0]. Padding is -INFINITY.
#include <cuda_runtime.h>
#include <math.h>

#define OAL_B 256

__global__ void rmax_part(const float *x, float *part, long long n)
{
    __shared__ float s[OAL_B];
    int l = threadIdx.x;
    long long i = (long long)blockIdx.x * OAL_B + l;
    s[l] = i < n ? x[i] : -INFINITY;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] = fmaxf(s[l], s[l + k]);
        __syncthreads();
    }
    if (l == 0)
        part[blockIdx.x] = s[0];
}

__global__ void rmax_fin(const float *part, float *y, long long nb)
{
    __shared__ float s[OAL_B];
    int l = threadIdx.x;
    float a = -INFINITY;
    for (long long j = l; j < nb; j += OAL_B)
        a = fmaxf(a, part[j]);
    s[l] = a;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] = fmaxf(s[l], s[l + k]);
        __syncthreads();
    }
    if (l == 0)
        y[0] = s[0];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0], nb = p[1];
    rmax_part<<<(unsigned)nb, OAL_B>>>((const float *)buf[0], (float *)buf[2], n);
    rmax_fin<<<1, OAL_B>>>((const float *)buf[2], (float *)buf[1], nb);
}
