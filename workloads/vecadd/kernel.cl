// SPDX-License-Identifier: Apache-2.0
__kernel void vecadd(__global const float *a, __global const float *b, __global float *y, const int n)
{
    int i = get_global_id(0);
    if (i < n)
        y[i] = a[i] + b[i];
}
