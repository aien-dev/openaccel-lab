// SPDX-License-Identifier: Apache-2.0
// Launch 1: each block of 256 scans its slice in shared memory (Hillis-Steele), writes the
//           block-local inclusive prefix to y and the block total to bsum[g].
// Launch 2: one block scans bsum in place (chunks of 256 with a running carry).
// Launch 3: every element of block g > 0 adds bsum[g - 1].
#include <cuda_runtime.h>

#define OAL_B 256

__global__ void scan_blk(const int *x, int *y, int *bsum, long long n)
{
    __shared__ int s[OAL_B];
    int l = threadIdx.x;
    long long i = (long long)blockIdx.x * OAL_B + l;
    s[l] = i < n ? x[i] : 0;
    __syncthreads();
    for (int off = 1; off < OAL_B; off <<= 1) {
        int t = l >= off ? s[l - off] : 0;
        __syncthreads();
        s[l] += t;
        __syncthreads();
    }
    if (i < n)
        y[i] = s[l];
    if (l == OAL_B - 1)
        bsum[blockIdx.x] = s[l];
}

__global__ void scan_sums(int *bsum, long long nb)
{
    __shared__ int s[OAL_B];
    int l = threadIdx.x;
    int carry = 0;
    for (long long base = 0; base < nb; base += OAL_B) {
        long long j = base + l;
        s[l] = j < nb ? bsum[j] : 0;
        __syncthreads();
        for (int off = 1; off < OAL_B; off <<= 1) {
            int t = l >= off ? s[l - off] : 0;
            __syncthreads();
            s[l] += t;
            __syncthreads();
        }
        if (j < nb)
            bsum[j] = s[l] + carry;
        carry += s[OAL_B - 1];
        __syncthreads();
    }
}

__global__ void scan_add(int *y, const int *bsum, long long n)
{
    long long i = (long long)blockIdx.x * OAL_B + threadIdx.x;
    if (blockIdx.x > 0 && i < n)
        y[i] += bsum[blockIdx.x - 1];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0], nb = p[1];
    int *x = (int *)buf[0], *y = (int *)buf[1], *bsum = (int *)buf[2];
    scan_blk<<<(unsigned)nb, OAL_B>>>(x, y, bsum, n);
    scan_sums<<<1, OAL_B>>>(bsum, nb);
    scan_add<<<(unsigned)nb, OAL_B>>>(y, bsum, n);
}
