// SPDX-License-Identifier: Apache-2.0
__kernel void silu(__global const float *x, __global float *y, const int n)
{
    int i = get_global_id(0);
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + exp(-v));
    }
}
