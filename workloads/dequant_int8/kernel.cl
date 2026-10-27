// SPDX-License-Identifier: Apache-2.0
__kernel void dequant_int8(__global const char *q, __global const float *s, __global float *y,
                           const int R, const int C)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= R || c >= C)
        return;
    y[r * C + c] = (float)q[r * C + c] * s[r];
}
