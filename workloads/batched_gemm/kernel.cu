// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TS 16

// One block per 16x16 tile of one batch member (blockIdx.z is the batch index). Tiles of A and B are
// staged in static shared memory; out-of-range tile elements load as 0.0f, adding exact zeros.
__global__ void batched_gemm(const float *A, const float *B, float *C, int M, int N, int K)
{
    __shared__ float As[TS][TS];
    __shared__ float Bs[TS][TS];
    long long bt = blockIdx.z;
    const float *Ab = A + bt * M * K;
    const float *Bb = B + bt * K * N;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TS + ty, col = blockIdx.x * TS + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        As[ty][tx] = (row < M && k0 + tx < K) ? Ab[(long long)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? Bb[(long long)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[(bt * M + row) * N + col] = acc;
}

void oal_launch(const long long *p, void *const *buf)
{
    int Bt = (int)p[0], M = (int)p[1], N = (int)p[2], K = (int)p[3];
    dim3 grid((unsigned)((N + TS - 1) / TS), (unsigned)((M + TS - 1) / TS), (unsigned)Bt);
    dim3 block(TS, TS);
    batched_gemm<<<grid, block>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], M, N, K);
}
