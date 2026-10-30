// SPDX-License-Identifier: Apache-2.0
// CONTROL (deliberately wrong): scans x shifted right by one, i.e. the EXCLUSIVE prefix sum.
// Launch 1: each work-group of 256 scans its slice in local memory (Hillis-Steele), writes the
//           block-local inclusive prefix to y and the block total to bsum[g].
// Launch 2: one work-group scans bsum in place (chunks of 256 with a running carry).
// Launch 3: every element of block g > 0 adds bsum[g - 1].

static void scan_tree(__local int *s, int l)
{
    for (int off = 1; off < (int)get_local_size(0); off <<= 1) {
        int t = l >= off ? s[l - off] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        s[l] += t;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void scan_blk(__global const int *x, __global int *y, __global int *bsum, const int n, __local int *s)
{
    int l = get_local_id(0);
    int i = get_global_id(0);
    s[l] = (i >= 1 && i < n) ? x[i - 1] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    scan_tree(s, l);
    if (i < n)
        y[i] = s[l];
    if (l == get_local_size(0) - 1)
        bsum[get_group_id(0)] = s[l];
}

__kernel void scan_sums(__global int *bsum, const int nb, __local int *s)
{
    int l = get_local_id(0);
    int w = get_local_size(0);
    int carry = 0;
    for (int base = 0; base < nb; base += w) {
        int j = base + l;
        s[l] = j < nb ? bsum[j] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);
        scan_tree(s, l);
        if (j < nb)
            bsum[j] = s[l] + carry;
        carry += s[w - 1];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void scan_add(__global int *y, __global const int *bsum, const int n)
{
    int i = get_global_id(0);
    int g = get_group_id(0);
    if (g > 0 && i < n)
        y[i] += bsum[g - 1];
}
