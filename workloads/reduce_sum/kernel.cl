// SPDX-License-Identifier: Apache-2.0
// Pass 1: each work-group of 256 sums its slice in a local-memory tree and writes one partial.
// Pass 2: one work-group sums the nb partials (strided, then a tree) into y[0].
__kernel void rsum_part(__global const float *x, __global float *part, const int n, __local float *s)
{
    int l = get_local_id(0);
    int i = get_global_id(0);
    s[l] = i < n ? x[i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        part[get_group_id(0)] = s[0];
}

__kernel void rsum_fin(__global const float *part, __global float *y, const int nb, __local float *s)
{
    int l = get_local_id(0);
    float a = 0.0f;
    for (int j = l; j < nb; j += get_local_size(0))
        a += part[j];
    s[l] = a;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        y[0] = s[0];
}
