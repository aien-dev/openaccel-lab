// SPDX-License-Identifier: Apache-2.0
// one work-item per rotated pair; n = T*H*h pairs
__kernel void rope(__global const float *x, __global const float *cs, __global const float *sn,
                   __global float *y, const int n, const int h, const int H)
{
    int g = get_global_id(0);
    if (g < n) {
        int i = g % h;
        int th = g / h;             // t*H + j
        int t = th / H;
        int o = th * 2 * h + i;
        float c = cs[t * h + i], s = sn[t * h + i];
        float x1 = x[o], x2 = x[o + h];
        y[o] = x1 * c - x2 * s;
        y[o + h] = x1 * s + x2 * c;
    }
}
