// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TS 64

// One thread per row; x is staged through static shared memory in chunks of 64 shared by the block.
// Every thread reaches every __syncthreads().
__global__ void gemv(const float *A, const float *x, float *y, int M, int N)
{
    __shared__ float xs[TS];
    int i = blockIdx.x * TS + threadIdx.x, lid = threadIdx.x;
    float acc = 0.0f;
    for (int j0 = 0; j0 < N; j0 += TS) {
        xs[lid] = (j0 + lid < N) ? x[j0 + lid] : 0.0f;
        __syncthreads();
        if (i < M) {
            int jn = N - j0 < TS ? N - j0 : TS;
            for (int j = 0; j < jn; ++j)
                acc += A[(long long)i * N + j0 + j] * xs[j];
        }
        __syncthreads();
    }
    if (i < M)
        y[i] = acc;
}

void oal_launch(const long long *p, void *const *buf)
{
    int M = (int)p[0], N = (int)p[1];
    gemv<<<(unsigned)((M + TS - 1) / TS), TS>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], M, N);
}
