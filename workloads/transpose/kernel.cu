// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

#define TS 16

// 16x16 tile through static shared memory (one padding column against bank conflicts).
__global__ void transpose(const float *a, float *b, int R, int C)
{
    __shared__ float tile[TS][TS + 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int c0 = blockIdx.x * TS, r0 = blockIdx.y * TS;
    int r = r0 + ty, c = c0 + tx;
    if (r < R && c < C)
        tile[ty][tx] = a[(long long)r * C + c];
    __syncthreads();
    int orow = c0 + ty, ocol = r0 + tx; // output b is C x R
    if (orow < C && ocol < R)
        b[(long long)orow * R + ocol] = tile[tx][ty];
}

void oal_launch(const long long *p, void *const *buf)
{
    int R = (int)p[0], C = (int)p[1];
    dim3 grid((unsigned)((C + TS - 1) / TS), (unsigned)((R + TS - 1) / TS));
    dim3 block(TS, TS);
    transpose<<<grid, block>>>((const float *)buf[0], (float *)buf[1], R, C);
}
