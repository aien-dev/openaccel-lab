// SPDX-License-Identifier: Apache-2.0
// One work-group of 256 per 16x16 tile; A and B are staged in local memory, C[i][j] = sum_k A[i][k] B[k][j].
__kernel void tmatmul16(__global const float *a, __global const float *b, __global float *c, const int tiles)
{
    __local float la[256];
    __local float lb[256];
    int t = get_group_id(0);
    int l = get_local_id(0);
    int i = l / 16, j = l % 16;
    la[l] = a[t * 256 + l];
    lb[l] = b[t * 256 + l];
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = 0.0f;
    for (int k = 0; k < 15; k++)
        s += la[i * 16 + k] * lb[k * 16 + j];
    c[t * 256 + l] = s;
}
