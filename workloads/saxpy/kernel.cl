// SPDX-License-Identifier: Apache-2.0
__kernel void saxpy(__global const float *alpha, __global const float *x, __global const float *y_in,
                    __global float *y, const int n)
{
    int i = get_global_id(0);
    if (i < n)
        y[i] = alpha[0] * x[i] + y_in[i];
}
