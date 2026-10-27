// SPDX-License-Identifier: Apache-2.0
// Pass 1: each work-group of 256 forms a[i]*b[i] and sums its slice in a local-memory tree.
// Pass 2: one work-group sums the nb partials (strided, then a tree) into y[0].
__kernel void dot_part(__global const float *a, __global const float *b, __global float *part, const int n,
                       __local float *s)
{
    int l = get_local_id(0);
    int i = get_global_id(0);
    s[l] = i < n ? a[i] * b[i] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        part[get_group_id(0)] = s[0];
}

__kernel void dot_fin(__global const float *part, __global float *y, const int nb, __local float *s)
{
    int l = get_local_id(0);
    float acc = 0.0f;
    for (int j = l; j < nb; j += get_local_size(0))
        acc += part[j];
    s[l] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k)
            s[l] += s[l + k];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (l == 0)
        y[0] = s[0];
}
