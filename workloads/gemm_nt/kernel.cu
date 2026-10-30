// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TS 16

// C = A B^T. 16x16 tiles of A and of B^T staged in static shared memory; out-of-range tile elements
// load as 0.0f, which adds exact zeros to the accumulator.
__global__ void gemm_nt(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS]; // Bs[k][n] = B[n][k]
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
    int bn = blockIdx.x * TS + ty; // B row this thread stages
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(long long)row * K + k0 + tx] : 0.0f;
        Bs[tx][ty] = (bn < N && k0 + tx < K) ? B[(long long)bn * K + k0 + tx] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[(long long)row * N + col] = acc;
}

void oal_launch(const long long *p, void *const *buf)
{
    int M = (int)p[0], N = (int)p[1], K = (int)p[2];
    dim3 grid((unsigned)((N + TS - 1) / TS), (unsigned)((M + TS - 1) / TS));
    dim3 block(TS, TS);
    gemm_nt<<<grid, block>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], M, N, K);
}
