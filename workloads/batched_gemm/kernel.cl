// SPDX-License-Identifier: Apache-2.0
// One work-group per 16x16 tile of one batch member (dimension 2 is the batch index). Tiles of A and B
// are staged in __local memory; out-of-range tile elements load as 0.0f, adding exact zeros.
#define TS 16
__kernel void batched_gemm(__global const float *A, __global const float *B, __global float *C,
                           const int M, const int N, const int K)
{
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    int bt = get_global_id(2);
    __global const float *Ab = A + bt * M * K;
    __global const float *Bb = B + bt * K * N;
    int tx = get_local_id(0), ty = get_local_id(1);
    int row = get_group_id(1) * TS + ty, col = get_group_id(0) * TS + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        As[ty][tx] = (row < M && k0 + tx < K) ? Ab[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? Bb[(k0 + ty) * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N)
        C[(bt * M + row) * N + col] = acc;
}
