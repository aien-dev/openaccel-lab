// SPDX-License-Identifier: Apache-2.0
// One work-item per row; x is staged through __local memory in chunks of 64 shared by the work-group.
// global is M rounded up to 64, local 64; every work-item reaches every barrier.
#define TS 64
__kernel void gemv(__global const float *A, __global const float *x, __global float *y, const int M, const int N)
{
    __local float xs[TS];
    int i = get_global_id(0), lid = get_local_id(0);
    float acc = 0.0f;
    for (int j0 = 0; j0 < N; j0 += TS) {
        xs[lid] = (j0 + lid < N) ? x[j0 + lid] : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (i < M) {
            int jn = min(TS, N - j0);
            for (int j = 0; j < jn; ++j)
                acc += A[i * N + j0 + j] * xs[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < M)
        y[i] = acc;
}
