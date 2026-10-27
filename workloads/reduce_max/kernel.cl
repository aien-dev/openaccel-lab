// SPDX-License-Identifier: Apache-2.0
// Pass 1: each work-group of 256 takes the max of its slice in a local-memory tree.
// Pass 2: one work-group takes the max of the nb partials into y[0]. Padding is -INFINITY.
__kernel void rmax_part(__global const float *x, __global float *part, const int n, __local float *s)
{
    int l = get_local_id(0);
    int i = get_global_id(0);
    s[l] = i < n ? x[i] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] = fmax(s[l], s[l + k]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        part[get_group_id(0)] = s[0];
}

__kernel void rmax_fin(__global const float *part, __global float *y, const int nb, __local float *s)
{
    int l = get_local_id(0);
    float a = -INFINITY;
    for (int j = l; j < nb; j += get_local_size(0))
        a = fmax(a, part[j]);
    s[l] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] = fmax(s[l], s[l + k]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        y[0] = s[0];
}
