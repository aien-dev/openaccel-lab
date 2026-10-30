// SPDX-License-Identifier: Apache-2.0
// (value, index) pairs; (v2, i2) beats (v1, i1) when v2 > v1, or v2 == v1 and i2 < i1 (first maximum).
// Pass 1: each work-group of 256 reduces its slice in local memory. Pass 2: one work-group reduces
// the nb partials. Padding is (-INFINITY, INT_MAX), which never beats a real element.

#define OAL_BEATS(v2, i2, v1, i1) ((v2) > (v1) || ((v2) == (v1) && (i2) < (i1)))

static void amax_tree(__local float *sv, __local int *si, int l)
{
    for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
        if (l < k && OAL_BEATS(sv[l + k], si[l + k], sv[l], si[l])) {
            sv[l] = sv[l + k];
            si[l] = si[l + k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void amax_part(__global const float *x, __global float *pval, __global int *pidx, const int n,
                        __local float *sv, __local int *si)
{
    int l = get_local_id(0);
    int i = get_global_id(0);
    sv[l] = i < n ? x[i] : -INFINITY;
    si[l] = i < n ? i : INT_MAX;
    barrier(CLK_LOCAL_MEM_FENCE);
    amax_tree(sv, si, l);
    if (l == 0) {
        pval[get_group_id(0)] = sv[0];
        pidx[get_group_id(0)] = si[0];
    }
}

__kernel void amax_fin(__global const float *pval, __global const int *pidx, __global int *y, const int nb,
                       __local float *sv, __local int *si)
{
    int l = get_local_id(0);
    float bv = -INFINITY;
    int bi = INT_MAX;
    for (int j = l; j < nb; j += get_local_size(0)) {
        if (OAL_BEATS(pval[j], pidx[j], bv, bi)) {
            bv = pval[j];
            bi = pidx[j];
        }
    }
    sv[l] = bv;
    si[l] = bi;
    barrier(CLK_LOCAL_MEM_FENCE);
    amax_tree(sv, si, l);
    if (l == 0)
        y[0] = si[0];
}
