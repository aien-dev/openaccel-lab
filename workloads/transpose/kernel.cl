// SPDX-License-Identifier: Apache-2.0
// 16x16 tile through __local memory (one padding column against bank conflicts); global is the
// tile-rounded (C, R) grid, local (16, 16).
#define TS 16
__kernel void transpose(__global const float *a, __global float *b, const int R, const int C)
{
    __local float tile[TS][TS + 1];
    int tx = get_local_id(0), ty = get_local_id(1);
    int c0 = get_group_id(0) * TS, r0 = get_group_id(1) * TS;
    int r = r0 + ty, c = c0 + tx;
    if (r < R && c < C)
        tile[ty][tx] = a[r * C + c];
    barrier(CLK_LOCAL_MEM_FENCE);
    int orow = c0 + ty, ocol = r0 + tx; // output b is C x R
    if (orow < C && ocol < R)
        b[orow * R + ocol] = tile[tx][ty];
}
