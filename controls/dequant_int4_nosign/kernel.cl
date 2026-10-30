// SPDX-License-Identifier: Apache-2.0
// CONTROL dequant_int4_nosign: the nibble is not sign-extended.
__kernel void dequant_int4(__global const uchar *w, __global const float *s, __global float *y,
                           const int R, const int C, const int G, const int NB)
{
    int r = get_global_id(0);
    int c = get_global_id(1);
    if (r >= R || c >= C)
        return;
    int b = w[r * NB + c / 2];
    int nib = (c & 1) ? (b >> 4) : (b & 15);
    y[r * C + c] = (float)nib * s[r * G + c / 32];
}
