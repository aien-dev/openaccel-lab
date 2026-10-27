// SPDX-License-Identifier: Apache-2.0
// (value, index) pairs; (v2, i2) beats (v1, i1) when v2 > v1, or v2 == v1 and i2 < i1 (first maximum).
// Pass 1: each block of 256 reduces its slice in shared memory. Pass 2: one block reduces the nb
// partials. Padding is (-INFINITY, INT_MAX), which never beats a real element.
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>

#define OAL_B 256
#define OAL_BEATS(v2, i2, v1, i1) ((v2) > (v1) || ((v2) == (v1) && (i2) < (i1)))

__global__ void amax_part(const float *x, float *pval, int *pidx, long long n)
{
    __shared__ float sv[OAL_B];
    __shared__ int si[OAL_B];
    int l = threadIdx.x;
    long long i = (long long)blockIdx.x * OAL_B + l;
    sv[l] = i < n ? x[i] : -INFINITY;
    si[l] = i < n ? (int)i : INT_MAX;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k && OAL_BEATS(sv[l + k], si[l + k], sv[l], si[l])) {
            sv[l] = sv[l + k];
            si[l] = si[l + k];
        }
        __syncthreads();
    }
    if (l == 0) {
        pval[blockIdx.x] = sv[0];
        pidx[blockIdx.x] = si[0];
    }
}

__global__ void amax_fin(const float *pval, const int *pidx, int *y, long long nb)
{
    __shared__ float sv[OAL_B];
    __shared__ int si[OAL_B];
    int l = threadIdx.x;
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (long long j = l; j < nb; j += OAL_B) {
        if (OAL_BEATS(pval[j], pidx[j], bv, bi)) {
            bv = pval[j];
            bi = pidx[j];
        }
    }
    sv[l] = bv;
    si[l] = bi;
    __syncthreads();
    for (int k = OAL_B / 2; k > 0; k >>= 1) {
        if (l < k && OAL_BEATS(sv[l + k], si[l + k], sv[l], si[l])) {
            sv[l] = sv[l + k];
            si[l] = si[l + k];
        }
        __syncthreads();
    }
    if (l == 0)
        y[0] = si[0];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0], nb = p[1];
    amax_part<<<(unsigned)nb, OAL_B>>>((const float *)buf[0], (float *)buf[2], (int *)buf[3], n);
    amax_fin<<<1, OAL_B>>>((const float *)buf[2], (const int *)buf[3], (int *)buf[1], nb);
}
