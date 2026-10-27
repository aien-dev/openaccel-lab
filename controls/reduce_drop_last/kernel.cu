// SPDX-License-Identifier: Apache-2.0
// CONTROL (deliberately wrong): pass 1 reads x[0..n-2] only, dropping the last element.
// Pass 1: each block of 256 sums its slice in a shared-memory tree and writes one partial.
// Pass 2: one block sums the nb partials (strided, then a tree) into y[0].
#include <cuda_runtime.h>

#define OAL_B 256

__global__ void rsum_part(const float *x, float *part, long long n)
{
    __shared__ float s[OAL_B];
    int l = threadIdx.x;
    long long i = (long long)blockIdx.x * OAL_B + l;
    s[l] = i < n - 1 ? x[i] : 0.0f;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        __syncthreads();
    }
    if (l == 0)
        part[blockIdx.x] = s[0];
}

__global__ void rsum_fin(const float *part, float *y, long long nb)
{
    __shared__ float s[OAL_B];
    int l = threadIdx.x;
    float a = 0.0f;
    for (long long j = l; j < nb; j += OAL_B)
        a += part[j];
    s[l] = a;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        __syncthreads();
    }
    if (l == 0)
        y[0] = s[0];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0], nb = p[1];
    rsum_part<<<(unsigned)nb, OAL_B>>>((const float *)buf[0], (float *)buf[2], n);
    rsum_fin<<<1, OAL_B>>>((const float *)buf[2], (float *)buf[1], nb);
}
