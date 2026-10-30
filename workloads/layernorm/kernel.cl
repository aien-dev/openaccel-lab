// SPDX-License-Identifier: Apache-2.0
// one work-group per row; get_local_size(0) must be a power of two
__kernel void layernorm(__global const float *x, __global const float *g, __global const float *b,
                        __global float *y, const int D, __local float *red)
{
    int t = get_local_id(0), nt = get_local_size(0);
    __global const float *xr = x + (long)get_group_id(0) * D;
    __global float *yr = y + (long)get_group_id(0) * D;
    float acc = 0.0f;
    for (int k = t; k < D; k += nt)
        acc += xr[k];
    red[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = nt / 2; h > 0; h >>= 1) {
        if (t < h)
            red[t] += red[t + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mu = red[0] / (float)D;
    barrier(CLK_LOCAL_MEM_FENCE);
    acc = 0.0f;
    for (int k = t; k < D; k += nt) {
        float c = xr[k] - mu;
        acc += c * c;
    }
    red[t] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int h = nt / 2; h > 0; h >>= 1) {
        if (t < h)
            red[t] += red[t + h];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float rs = rsqrt(red[0] / (float)D + 1e-5f);
    for (int k = t; k < D; k += nt)
        yr[k] = (xr[k] - mu) * rs * g[k] + b[k];
}
