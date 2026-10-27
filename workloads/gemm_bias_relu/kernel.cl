// SPDX-License-Identifier: Apache-2.0
// C = relu(A B + bias). 16x16 tiles of A and B staged in __local memory; out-of-range tile elements
// load as 0.0f, which adds exact zeros to the accumulator. global is the tile-rounded (N, M) grid,
// local (16, 16).
#define TS 16
__kernel void gemm_bias_relu(__global const float *A, __global const float *B, __global const float *bias,
                             __global float *C, const int M, const int N, const int K)
{
    __local float As[TS][TS];
    __local float Bs[TS][TS];
    int tx = get_local_id(0), ty = get_local_id(1);
    int row = get_group_id(1) * TS + ty, col = get_group_id(0) * TS + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; ++k)
            acc += As[ty][k] * Bs[k][tx];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (row < M && col < N)
        C[row * N + col] = fmax(acc + bias[col], 0.0f);
}
